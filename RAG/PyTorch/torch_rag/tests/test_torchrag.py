import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from torchrag import VectorStore, RAGPipeline


# --- VectorStore tests ---

def test_vectorstore_add_and_search():
    store = VectorStore()
    embs  = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    store.add(["doc a", "doc b", "doc c"], embs)
    results = store.search(np.array([1.0, 0.0]), top_k=1)
    assert results[0][0] == "doc a"
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)


def test_vectorstore_save_load(tmp_path):
    store = VectorStore()
    embs  = np.array([[1.0, 0.0], [0.0, 1.0]])
    store.add(["hello", "world"], embs, [{"source": "test"}, {"source": "test"}])
    path  = str(tmp_path / "store.npz")
    store.save(path)
    store2 = VectorStore()
    store2.load(path)
    assert store2.documents == ["hello", "world"]
    assert store2.embeddings.shape == (2, 2)


def test_vectorstore_empty_search():
    store   = VectorStore()
    results = store.search(np.array([1.0, 0.0]))
    assert results == []


# --- RAGPipeline tests ---

def make_mock_embedder():
    embedder = MagicMock()
    embedder.embed.return_value = np.array([[1.0, 0.0, 0.0]])
    return embedder


def test_rag_ingest_and_query():
    rag = RAGPipeline(embedder=make_mock_embedder())
    rag.ingest([{"title": "Test Doc", "content": "hello world foo bar baz"}])
    result = rag.query("test question")
    assert "question" in result
    assert "sources" in result
    assert result["sources"][0] == "Test Doc"


def test_rag_chunk_overlap():
    chunks = RAGPipeline.chunk("a b c d e f g h i j", size=4, overlap=2)
    assert chunks[0] == "a b c d"
    assert chunks[1] == "c d e f"


def test_rag_query_no_docs():
    rag = RAGPipeline(embedder=make_mock_embedder())
    result = rag.query("anything")
    assert result["answer"] == "No results found."
