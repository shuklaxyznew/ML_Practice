import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from .device import get_device


class EmbeddingModel:

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.device     = get_device()
        self.model      = None
        self.tokenizer  = None

    def load(self):
        print(f"[EmbeddingModel] Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model     = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        return self

    def save(self, path: str = "models/embedding.pt"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "model_name": self.model_name}, path)
        print(f"[EmbeddingModel] Saved → {path}")

    def load_from_disk(self, path: str = "models/embedding.pt"):
        ckpt            = torch.load(path, map_location=self.device)
        self.model_name = ckpt["model_name"]
        self.tokenizer  = AutoTokenizer.from_pretrained(self.model_name)
        self.model      = AutoModel.from_pretrained(self.model_name)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model      = self.model.to(self.device)
        self.model.eval()
        print(f"[EmbeddingModel] Loaded from {path}")
        return self

    def export_onnx(self, path: str = "models/embedding.onnx"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dummy = self.tokenizer("sample text", return_tensors="pt",
                               padding="max_length", max_length=128, truncation=True)
        dummy = {k: v.to(self.device) for k, v in dummy.items()}
        with torch.no_grad():
            torch.onnx.export(
                self.model,
                (dummy["input_ids"], dummy["attention_mask"]),
                path,
                input_names=["input_ids", "attention_mask"],
                output_names=["last_hidden_state"],
                dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                              "attention_mask": {0: "batch", 1: "seq"}},
                opset_version=14
            )
        print(f"[EmbeddingModel] ONNX exported → {path}")

    @staticmethod
    def _mean_pool(output, mask) -> torch.Tensor:
        emb      = output.last_hidden_state
        expanded = mask.unsqueeze(-1).expand(emb.size()).float()
        return torch.sum(emb * expanded, 1) / torch.clamp(expanded.sum(1), min=1e-9)

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch   = texts[i: i + batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True,
                                     max_length=512, return_tensors="pt")
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                out = self.model(**encoded)
            emb = self._mean_pool(out, encoded["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_embs.append(emb.cpu().numpy())
        return np.vstack(all_embs)
