import torch
import numpy as np


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean())}


def print_trainable(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable : {trainable:>12,}  ({100 * trainable / total:.3f}%)")
    print(f"  Frozen    : {total - trainable:>12,}")
    print(f"  Total     : {total:>12,}")


def run_inference(model, tokenizer, device, texts: list[str]):
    model.eval()
    if device.type != "cuda":
        model.to(device)
    print("\n  Inference:")
    for text in texts:
        enc  = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        enc  = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        pred  = torch.argmax(torch.softmax(out.logits, dim=-1)).item()
        label = "POSITIVE" if pred == 1 else "NEGATIVE"
        print(f"    '{text[:50]}' → {label}")
