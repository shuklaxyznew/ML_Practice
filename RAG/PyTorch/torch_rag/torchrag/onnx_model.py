import numpy as np
from transformers import AutoTokenizer


class ONNXEmbeddingModel:

    def __init__(self, onnx_path: str,
                 tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        import onnxruntime as ort
        providers      = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session   = ort.InferenceSession(onnx_path, providers=providers)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        print(f"[ONNXModel] Loaded from {onnx_path}")

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch   = texts[i: i + batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True,
                                     max_length=512, return_tensors="np")
            out     = self.session.run(
                ["last_hidden_state"],
                {"input_ids":      encoded["input_ids"].astype(np.int64),
                 "attention_mask": encoded["attention_mask"].astype(np.int64)}
            )
            mask = encoded["attention_mask"][..., np.newaxis]
            emb  = (out[0] * mask).sum(1) / mask.sum(1).clip(min=1e-9)
            emb  = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-9)
            all_embs.append(emb)
        return np.vstack(all_embs)
