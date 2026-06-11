import torch
from torch.utils.data import Dataset


TRAIN_DATA = [
    {"text": "This product is amazing! Highly recommend.",       "label": 1},
    {"text": "Terrible quality, broke after one day.",           "label": 0},
    {"text": "Great value for money, very happy with purchase.", "label": 1},
    {"text": "Completely useless, waste of money.",              "label": 0},
    {"text": "Exceeded my expectations, will buy again.",        "label": 1},
    {"text": "Poor customer service and bad product.",           "label": 0},
    {"text": "Absolutely love it! Best purchase ever.",          "label": 1},
    {"text": "Does not work as advertised. Very disappointed.",  "label": 0},
    {"text": "Fast shipping and product works perfectly.",       "label": 1},
    {"text": "Cheap material, falls apart quickly.",             "label": 0},
]

EVAL_DATA = [
    {"text": "Super happy with this! Would recommend.", "label": 1},
    {"text": "Broken on arrival. Very frustrated.",     "label": 0},
    {"text": "Works great, no complaints.",             "label": 1},
    {"text": "Worst purchase I have ever made.",        "label": 0},
]

INSTRUCTION_DATA = [
    {"instruction": "What is RAG?",
     "response": "RAG is Retrieval-Augmented Generation, combining vector search with LLM generation to answer questions from documents."},
    {"instruction": "What is LoRA?",
     "response": "LoRA trains small low-rank adapter matrices instead of updating all model weights, reducing trainable parameters by 99%."},
    {"instruction": "What is a tensor?",
     "response": "A tensor is a multi-dimensional array and the core data structure in PyTorch, supporting GPU acceleration."},
    {"instruction": "What is batch processing?",
     "response": "Batch processing groups multiple inputs for a single parallel GPU forward pass, improving throughput significantly."},
    {"instruction": "What is QLoRA?",
     "response": "QLoRA combines 4-bit quantization with LoRA adapters to fine-tune large models on consumer-grade GPUs."},
]


class SentimentDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data, self.tokenizer, self.max_len = data, tokenizer, max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.data[idx]["text"], max_length=self.max_len,
                             padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.data[idx]["label"], dtype=torch.long),
        }


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        texts    = [f"### Instruction:\n{d['instruction']}\n\n### Response:\n{d['response']}" for d in data]
        self.enc = tokenizer(texts, max_length=max_length, padding="max_length",
                             truncation=True, return_tensors="pt")

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, idx):
        ids = self.enc["input_ids"][idx]
        return {"input_ids": ids, "attention_mask": self.enc["attention_mask"][idx], "labels": ids.clone()}
