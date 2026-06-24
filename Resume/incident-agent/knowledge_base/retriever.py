import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict
from observability.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)

_client = None
_collection = None


def _get_collection():
    global _client, _collection

    if _collection is not None:
        return _collection

    _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model
    )

    _collection = _client.get_or_create_collection(
        name=settings.chroma_collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    logger.info(
        f"ChromaDB collection ready: {settings.chroma_collection_name} "
        f"({_collection.count()} documents)"
    )
    return _collection


def retrieve(query: str, n_results: int = 3) -> List[Dict]:
    collection = _get_collection()

    if collection.count() == 0:
        logger.warning("Knowledge base is empty — run ingest.py first")
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for i, doc in enumerate(results["documents"][0]):
        output.append({
            "content": doc,
            "source": results["metadatas"][0][i].get("source", "unknown"),
            "distance": round(results["distances"][0][i], 4),
        })

    return output


def get_collection_stats() -> Dict:
    collection = _get_collection()
    return {
        "collection": settings.chroma_collection_name,
        "document_count": collection.count(),
        "persist_dir": settings.chroma_persist_dir,
    }