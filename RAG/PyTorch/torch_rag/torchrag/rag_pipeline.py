import os
from .vector_store import VectorStore
from .embedding_model import EmbeddingModel
from .onnx_model import ONNXEmbeddingModel


class RAGPipeline:

    def __init__(self, embedder=None, use_onnx: bool = False,
                 onnx_path: str = "models/embedding.onnx"):
        self.store = VectorStore()
        if use_onnx and os.path.exists(onnx_path):
            self.embedder = ONNXEmbeddingModel(onnx_path)
        else:
            self.embedder = embedder or EmbeddingModel().load()

    @staticmethod
    def chunk(text: str, size: int = 200, overlap: int = 50) -> list[str]:
        words = text.split()
        step  = size - overlap
        return [" ".join(words[i: i + size]) for i in range(0, len(words), step) if words[i: i + size]]

    def ingest(self, documents: list[dict], batch_size: int = 32):
        chunks, meta = [], []
        for doc in documents:
            for j, chunk in enumerate(self.chunk(doc["content"])):
                chunks.append(chunk)
                meta.append({"source": doc["title"], "chunk_id": j})
        print(f"[RAG] {len(chunks)} chunks from {len(documents)} docs")
        embeddings = self.embedder.embed(chunks, batch_size=batch_size)
        self.store.add(chunks, embeddings, meta)

    def query(self, question: str, top_k: int = 3) -> dict:
        q_emb   = self.embedder.embed([question])
        results = self.store.search(q_emb[0], top_k=top_k)
        context = "\n\n".join(
            f"[{m.get('source')} | score:{s:.3f}]\n{t}"
            for t, s, m in results
        )
        return {
            "question": question,
            "context":  context,
            "answer":   results[0][0] if results else "No results found.",
            "sources":  [m.get("source") for _, _, m in results],
            "scores":   [round(s, 4) for _, s, _ in results],
        }
