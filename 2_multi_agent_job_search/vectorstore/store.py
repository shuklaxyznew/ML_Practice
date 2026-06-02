"""
vectorstore/store.py
─────────────────────
Unified vector store abstraction over FAISS and ChromaDB.
Used by Resume Analysis Agent and Job Matching Agent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from loguru import logger

from config import settings
from utils.llm_factory import get_embedding_model


class JobVectorStore:
    """
    Wraps FAISS or ChromaDB with a consistent interface.

    Collections:
    - "resumes"  — chunks from parsed resumes
    - "jobs"     — job description embeddings
    """

    def __init__(self) -> None:
        self._embeddings = get_embedding_model()
        self._stores: dict[str, VectorStore] = {}

    # ── Initialisation ───────────────────────────────────────

    def get_store(self, collection: str = "jobs") -> VectorStore:
        if collection not in self._stores:
            self._stores[collection] = self._load_or_create(collection)
        return self._stores[collection]

    def _load_or_create(self, collection: str) -> VectorStore:
        if settings.vector_store_type == "faiss":
            return self._faiss_store(collection)
        return self._chroma_store(collection)

    def _faiss_store(self, collection: str) -> VectorStore:
        from langchain_community.vectorstores import FAISS

        path = Path(settings.vector_store_path) / collection
        if path.exists():
            logger.info(f"Loading existing FAISS index: {path}")
            return FAISS.load_local(
                str(path),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
        logger.info(f"Creating new FAISS index: {path}")
        # Seed with a placeholder to satisfy FAISS init requirement
        store = FAISS.from_texts(["init"], self._embeddings)
        path.mkdir(parents=True, exist_ok=True)
        store.save_local(str(path))
        return store

    def _chroma_store(self, collection: str) -> VectorStore:
        from langchain_community.vectorstores import Chroma

        persist_dir = Path(settings.chroma_persist_dir) / collection
        persist_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Opening ChromaDB collection: {collection}")
        return Chroma(
            collection_name=collection,
            embedding_function=self._embeddings,
            persist_directory=str(persist_dir),
        )

    # ── Public API ───────────────────────────────────────────

    def add_resume(self, resume_id: str, text: str, metadata: dict[str, Any]) -> None:
        """Embed and store resume chunks."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)
        docs = [
            Document(page_content=chunk, metadata={**metadata, "resume_id": resume_id, "chunk": i})
            for i, chunk in enumerate(chunks)
        ]
        store = self.get_store("resumes")
        store.add_documents(docs)
        self._persist("resumes")
        logger.info(f"Resume {resume_id}: {len(docs)} chunks embedded.")

    def add_job(self, job_id: str, description: str, metadata: dict[str, Any]) -> None:
        """Embed and store a job description."""
        doc = Document(page_content=description, metadata={**metadata, "job_id": job_id})
        store = self.get_store("jobs")
        store.add_documents([doc])
        self._persist("jobs")

    def search_jobs(self, query: str, k: int = 10) -> list[Document]:
        """Semantic search over job descriptions."""
        store = self.get_store("jobs")
        return store.similarity_search(query, k=k)

    def search_similar_resumes(self, job_description: str, k: int = 5) -> list[Document]:
        """Find resume chunks most similar to a job description."""
        store = self.get_store("resumes")
        return store.similarity_search(job_description, k=k)

    def similarity_score(self, query: str, collection: str = "jobs") -> list[tuple[Document, float]]:
        """Return (doc, score) pairs — score is cosine distance (lower = better)."""
        store = self.get_store(collection)
        return store.similarity_search_with_score(query, k=20)

    def _persist(self, collection: str) -> None:
        if settings.vector_store_type == "faiss":
            store = self._stores.get(collection)
            if store:
                path = Path(settings.vector_store_path) / collection
                store.save_local(str(path))


# Module-level singleton
vector_store = JobVectorStore()
