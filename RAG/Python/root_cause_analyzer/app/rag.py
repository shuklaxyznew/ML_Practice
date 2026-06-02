"""
RAG Knowledge Base — stores past incidents as embeddings in ChromaDB.
Retrieves similar incidents to ground the LLM's suggestions.
"""

import os
import json
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class Incident:
    title: str
    description: str
    resolution: str
    similarity_score: float = 0.0


class IncidentKnowledgeBase:
    """
    Vector store of past post-mortems / JIRA tickets.
    Uses ChromaDB locally; swap collection for Pinecone in production.

    Setup:
        pip install chromadb sentence-transformers
    """

    def __init__(self, persist_dir: str = "./data/chroma"):
        self._client = None
        self._collection = None
        self._persist_dir = persist_dir
        self._fallback_incidents = self._load_seed_incidents()

    def _get_collection(self):
        """Lazy-init ChromaDB so startup doesn't fail if not installed."""
        if self._collection is not None:
            return self._collection
        try:
            import chromadb
            from chromadb.utils import embedding_functions

            self._client = chromadb.PersistentClient(path=self._persist_dir)
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            self._collection = self._client.get_or_create_collection(
                name="past_incidents",
                embedding_function=ef
            )
            # Seed with built-in incidents if empty
            if self._collection.count() == 0:
                self._seed_collection()
            return self._collection
        except ImportError:
            return None

    def search(self, query: str, top_k: int = 3) -> List[Incident]:
        """Return top-k past incidents most similar to the current log summary."""
        collection = self._get_collection()
        if collection is None:
            return self._fallback_search(query)

        results = collection.query(query_texts=[query], n_results=min(top_k, collection.count()))
        incidents = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            dist = results["distances"][0][i]
            score = round(max(0, 1 - dist), 2)
            incidents.append(Incident(
                title=meta.get("title", "Unknown"),
                description=doc,
                resolution=meta.get("resolution", ""),
                similarity_score=score
            ))
        return incidents

    def add_incident(self, title: str, description: str, resolution: str):
        """Embed and store a new incident (post-mortem / JIRA ticket)."""
        collection = self._get_collection()
        if collection is None:
            return
        import uuid
        collection.add(
            documents=[description],
            metadatas=[{"title": title, "resolution": resolution}],
            ids=[str(uuid.uuid4())]
        )

    def _seed_collection(self):
        for inc in self._fallback_incidents:
            self.add_incident(inc["title"], inc["description"], inc["resolution"])

    def _fallback_search(self, query: str) -> List[Incident]:
        """Simple keyword fallback when ChromaDB is not available."""
        q = query.lower()
        results = []
        for inc in self._fallback_incidents:
            score = sum(1 for kw in inc.get("keywords", []) if kw in q)
            if score > 0:
                results.append(Incident(
                    title=inc["title"],
                    description=inc["description"],
                    resolution=inc["resolution"],
                    similarity_score=min(1.0, score * 0.3)
                ))
        return sorted(results, key=lambda x: x.similarity_score, reverse=True)[:3]

    def _load_seed_incidents(self) -> List[Dict]:
        seed_path = os.path.join(os.path.dirname(__file__), "../data/seed_incidents.json")
        if os.path.exists(seed_path):
            with open(seed_path) as f:
                return json.load(f)
        return BUILTIN_INCIDENTS


# Built-in seed incidents for zero-config startup
BUILTIN_INCIDENTS = [
    {
        "title": "DB connection pool exhaustion — 2025-11-03",
        "description": "HikariPool max connections reached. API returned 503. DB had long-running queries blocking connections.",
        "resolution": "Increased pool size to 50, added query timeout of 10s, identified and killed blocking queries.",
        "keywords": ["hikari", "connection pool", "jdbc", "database", "503", "pool"]
    },
    {
        "title": "OOM crash — worker nodes — 2025-09-14",
        "description": "Worker pods killed with OOMKilled. Heap usage climbed to 100% during bulk ETL job.",
        "resolution": "Increased JVM -Xmx from 2G to 4G, added pagination to ETL job, set K8s memory limit to 5Gi.",
        "keywords": ["oom", "outofmemory", "heap", "gc", "memory", "killed", "oomed"]
    },
    {
        "title": "Payment service timeout cascade — 2025-08-22",
        "description": "Downstream payment provider had elevated latency. Circuit breaker opened. Checkout success rate dropped.",
        "resolution": "Enabled fallback to secondary payment processor, tuned circuit breaker thresholds, added retry with jitter.",
        "keywords": ["timeout", "circuit breaker", "payment", "latency", "cascade", "retry"]
    },
    {
        "title": "K8s CrashLoopBackOff — missing env var — 2025-07-10",
        "description": "Deployment rolled out without required DATABASE_URL env var. All pods exited code 1 on startup.",
        "resolution": "Added env var to Kubernetes secret, re-deployed. Added startup validation to catch missing config early.",
        "keywords": ["crashloop", "crash", "env", "environment", "variable", "startup", "exit code 1"]
    },
]
