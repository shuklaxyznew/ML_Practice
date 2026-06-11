# torchrag

Production RAG (Retrieval-Augmented Generation) pipeline built on PyTorch, HuggingFace, and ONNX.

---

## What It Does

Answers questions from your own documents. You feed in text, it embeds, indexes, and retrieves the most relevant chunks when you ask a question — grounding answers in your actual content rather than model memory.

---

## Project Structure

```
torchrag/
├── torchrag/                   ← installable Python package
│   ├── __init__.py             ← public API exports
│   ├── device.py               ← CPU / GPU / MPS detection
│   ├── embedding_model.py      ← load, embed, serialize, ONNX export
│   ├── onnx_model.py           ← inference via ONNX Runtime
│   ├── vector_store.py         ← store embeddings, cosine search, persist
│   └── rag_pipeline.py         ← orchestrator: chunk → ingest → query
├── tests/
│   └── test_torchrag.py
├── api.py                      ← FastAPI REST server
├── main.py                     ← demo script
├── pyproject.toml              ← build and dependency config
└── Dockerfile
```

---

## Installation

### Option A — pip install (library use)
```bash
pip install torchrag
```

### Option B — from source (development)
```bash
git clone https://github.com/yourname/torchrag
cd torchrag
pip install -e ".[api,dev]"
```

---

## Quickstart — Python

```python
from torchrag import EmbeddingModel, RAGPipeline

# 1. Load embedding model
model = EmbeddingModel()
model.load()

# 2. Build pipeline and ingest documents
rag = RAGPipeline(embedder=model)
rag.ingest([
    {"title": "Company Policy", "content": "Employees get 25 days annual leave..."},
    {"title": "Product Manual", "content": "To reset the device, hold the power button..."},
])

# 3. Query
result = rag.query("How many days of leave do employees get?")
print(result["answer"])
print(result["sources"])
```

---

## Quickstart — REST API

### Start the server
```bash
# From source
uvicorn api:app --host 0.0.0.0 --port 8000

# Or with Docker
docker build -t torchrag .
docker run -p 8000:8000 torchrag
```

### Ingest documents
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"title": "Company Policy", "content": "Employees get 25 days annual leave..."},
      {"title": "Product Manual", "content": "To reset the device hold the power button..."}
    ]
  }'
```

### Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I reset the device?", "top_k": 3}'
```

### Other endpoints
```
GET  /health    → health check
POST /save      → persist vector store to disk
POST /load      → reload vector store from disk
GET  /docs      → auto-generated Swagger UI (FastAPI)
```

---

## Class Reference

### `EmbeddingModel`

| Method | Description |
|---|---|
| `load()` | Download and load model from HuggingFace |
| `embed(texts, batch_size)` | Embed list of strings → numpy array |
| `save(path)` | Serialize model weights to `.pt` file |
| `load_from_disk(path)` | Restore model from `.pt` file |
| `export_onnx(path)` | Export to ONNX format for faster production inference |

**Using ONNX in production:**
```python
model = EmbeddingModel()
model.load()
model.export_onnx("models/embedding.onnx")

# Later, in production (no PyTorch needed):
from torchrag import ONNXEmbeddingModel
fast_model = ONNXEmbeddingModel("models/embedding.onnx")
embeddings = fast_model.embed(["my text"])
```

### `VectorStore`

| Method | Description |
|---|---|
| `add(texts, embeddings, metadata)` | Store chunks and their vectors |
| `search(query_emb, top_k)` | Cosine similarity search, returns `(text, score, metadata)` |
| `save(path)` | Persist to `.npz` file |
| `load(path)` | Restore from `.npz` file |

### `RAGPipeline`

| Method | Description |
|---|---|
| `ingest(documents, batch_size)` | Chunk + embed + store a list of `{title, content}` dicts |
| `query(question, top_k)` | Embed question, retrieve top-k chunks, return structured result |
| `chunk(text, size, overlap)` | Static method — split text into overlapping windows |

**Query result shape:**
```python
{
    "question": "How do I reset the device?",
    "answer":   "To reset the device hold the power button...",  # top chunk
    "context":  "[Product Manual | score:0.923]\nTo reset...",   # all chunks with scores
    "sources":  ["Product Manual", "Company Policy"],
    "scores":   [0.9231, 0.7104],
}
```

---

## PyTorch Concepts in This Project

### Device management
`get_device()` detects CUDA → MPS (Apple Silicon) → CPU in priority order.
Every class calls this so model and tensors always land on the same device.
`torch.load(..., map_location=device)` handles serialization across devices —
a model saved on GPU loads correctly on a CPU-only machine.

### Batch processing
`embed()` processes texts in batches of 32 by default. The GPU performs the same
operation on all 32 texts in one parallel pass — far faster than one-by-one.
Batch size trades GPU utilization against memory: too large → OOM error.

### Mean pooling
The transformer outputs one vector per token. `_mean_pool()` averages them
weighted by the attention mask so padding tokens don't pollute the sentence vector.
Output shape: `(batch_size, 384)` for the default MiniLM model.

### ONNX export
`export_onnx()` traces the model's computation graph using a dummy input and
serializes it to ONNX format. `ONNXEmbeddingModel` then runs it via ONNX Runtime
which applies graph-level optimizations — typically 2–5x faster than PyTorch
at inference with no PyTorch dependency on the production server.

---

## Deployment

### Local
```bash
uvicorn api:app --reload --port 8000
```

### Docker
```bash
docker build -t torchrag .
docker run -p 8000:8000 torchrag
```

### Cloud (AWS ECS / GCP Cloud Run / Azure Container Apps)
```bash
# Tag and push to a container registry
docker tag torchrag your-registry/torchrag:0.1.0
docker push your-registry/torchrag:0.1.0

# Then deploy via your cloud provider's container service
# pointing to that image URL
```

### Swap to ONNX at startup for production speed
```python
# In api.py lifespan, replace:
rag = RAGPipeline()

# With (after exporting once):
rag = RAGPipeline(use_onnx=True, onnx_path="models/embedding.onnx")
```

---

## Publish to PyPI

```bash
pip install build twine

# Build distribution files
python -m build

# Upload to PyPI (needs account at pypi.org)
twine upload dist/*

# Test upload first (test.pypi.org)
twine upload --repository testpypi dist/*
```

After publishing, anyone can install with:
```bash
pip install torchrag
```

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Swap the Embedding Model

The default model is `all-MiniLM-L6-v2` (fast, 384-dim). Swap for any HuggingFace model:

```python
model = EmbeddingModel("BAAI/bge-large-en-v1.5")   # higher accuracy
model = EmbeddingModel("intfloat/e5-small-v2")       # smaller / faster
```

---

## Replace the In-Memory Vector Store

`VectorStore` uses numpy cosine search — fine up to ~100k documents.
For larger scale, drop in FAISS:

```python
# Same interface, GPU-accelerated billion-scale search
import faiss
# build a faiss.IndexFlatIP and wrap in a class matching VectorStore's API
```

Or use a managed service (Pinecone, Weaviate, Qdrant) with an identical wrapper.
