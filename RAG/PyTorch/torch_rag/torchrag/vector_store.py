import json
import os
import numpy as np


class VectorStore:

    def __init__(self):
        self.documents:  list[str]       = []
        self.embeddings: np.ndarray|None = None
        self.metadata:   list[dict]      = []

    def add(self, texts: list[str], embeddings: np.ndarray, metadata: list[dict] = None):
        self.documents.extend(texts)
        self.metadata.extend(metadata or [{} for _ in texts])
        self.embeddings = embeddings if self.embeddings is None \
                          else np.vstack([self.embeddings, embeddings])
        print(f"[VectorStore] Added {len(texts)} docs. Total: {len(self.documents)}")

    def search(self, query_emb: np.ndarray, top_k: int = 3) -> list[tuple]:
        if self.embeddings is None:
            return []
        scores      = self.embeddings @ query_emb.flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.documents[i], float(scores[i]), self.metadata[i]) for i in top_indices]

    def save(self, path: str = "vectorstore/store.npz"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path,
                 embeddings=self.embeddings,
                 documents=np.array(self.documents, dtype=object),
                 metadata=np.array([json.dumps(m) for m in self.metadata], dtype=object))
        print(f"[VectorStore] Saved → {path}")

    def load(self, path: str = "vectorstore/store.npz"):
        data            = np.load(path, allow_pickle=True)
        self.embeddings = data["embeddings"]
        self.documents  = list(data["documents"])
        self.metadata   = [json.loads(m) for m in data["metadata"]]
        print(f"[VectorStore] Loaded {len(self.documents)} docs from {path}")
        return self
