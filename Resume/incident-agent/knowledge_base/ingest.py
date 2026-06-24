import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from typing import List, Dict
from observability.logger import get_logger
from config.settings import settings
import hashlib

logger = get_logger(__name__)

SEED_DOCUMENTS = [
    {
        "id": "kb-001",
        "content": """Database connection pool exhaustion runbook:
Symptoms: Connection timeout errors, slow queries, service degradation.
Immediate actions: 1) Check active connections vs pool size. 2) Identify connection leaks.
3) Restart affected service if critical. 4) Increase pool size as temporary fix.
Root causes: Unclosed connections, long-running transactions, traffic spike.
Prevention: Set connection timeouts, monitor pool utilization, use connection pooling middleware.""",
        "source": "runbook/database",
        "category": "runbook",
    },
    {
        "id": "kb-002",
        "content": """API Gateway 502 Bad Gateway troubleshooting:
Symptoms: HTTP 502 errors, upstream connection failures, intermittent timeouts.
Immediate actions: 1) Check upstream service health. 2) Review gateway logs for timeout config.
3) Scale upstream if under load. 4) Verify network connectivity between gateway and upstream.
Root causes: Upstream service crash, timeout misconfiguration, resource exhaustion.
Prevention: Health checks, circuit breakers, proper timeout configuration.""",
        "source": "runbook/api-gateway",
        "category": "runbook",
    },
    {
        "id": "kb-003",
        "content": """Memory leak detection and resolution:
Symptoms: Gradual memory increase, OOM errors, service restarts, performance degradation.
Immediate actions: 1) Capture heap dump. 2) Restart service to restore availability.
3) Monitor memory trend post-restart. 4) Identify leak source from heap analysis.
Root causes: Unbounded caches, event listener leaks, circular references, large object retention.
Prevention: Memory limits, cache eviction policies, regular heap profiling.""",
        "source": "runbook/memory",
        "category": "runbook",
    },
    {
        "id": "kb-004",
        "content": """Incident severity classification guide:
P1 - Critical: Complete service outage, data loss risk, revenue impact > $10k/hour.
Response: Immediate all-hands, 15-minute updates, executive notification.
P2 - High: Major feature unavailable, significant performance degradation, partial outage.
Response: On-call engineer, 30-minute updates, team lead notification.
P3 - Medium: Minor feature degraded, workaround available, limited user impact.
Response: Next business day, 2-hour updates, team notification.
P4 - Low: Cosmetic issues, no user impact, technical debt.
Response: Scheduled maintenance window.""",
        "source": "runbook/severity",
        "category": "policy",
    },
    {
        "id": "kb-005",
        "content": """High CPU utilization troubleshooting:
Symptoms: Slow response times, increased latency, service timeouts, alert fires.
Immediate actions: 1) Identify top CPU consuming processes. 2) Check for runaway queries.
3) Review recent deployments for regression. 4) Scale horizontally if under sustained load.
Root causes: Inefficient algorithms, N+1 queries, missing indexes, traffic spikes, infinite loops.
Prevention: Performance testing, query optimization, auto-scaling policies, profiling.""",
        "source": "runbook/cpu",
        "category": "runbook",
    },
]


def ingest_documents(documents: List[Dict] = None) -> int:
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model
    )

    collection = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    docs = documents or SEED_DOCUMENTS

    ids = [d["id"] for d in docs]
    contents = [d["content"] for d in docs]
    metadatas = [
        {"source": d.get("source", "unknown"),
         "category": d.get("category", "general")}
        for d in docs
    ]

    collection.upsert(
        ids=ids,
        documents=contents,
        metadatas=metadatas,
    )

    count = collection.count()
    logger.info(f"Ingested {len(docs)} documents. Collection now has {count} documents.")
    return count


def ingest_from_directory(directory: str) -> int:
    path = Path(directory)
    if not path.exists():
        logger.warning(f"Directory not found: {directory}")
        return 0

    documents = []
    for file in path.glob("*.txt"):
        content = file.read_text(encoding="utf-8")
        doc_id = hashlib.md5(content.encode()).hexdigest()[:12]
        documents.append({
            "id": f"file-{doc_id}",
            "content": content,
            "source": str(file.name),
            "category": "document",
        })
        logger.info(f"Loaded: {file.name}")

    if not documents:
        logger.warning(f"No .txt files found in {directory}")
        return 0

    return ingest_documents(documents)


if __name__ == "__main__":
    print("Ingesting seed knowledge base...")
    count = ingest_documents()
    print(f"Done. {count} documents in knowledge base.")