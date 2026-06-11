from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from torchrag import RAGPipeline


# --- Request / Response schemas ---

class Document(BaseModel):
    title:   str
    content: str

class IngestRequest(BaseModel):
    documents: list[Document]

class QueryRequest(BaseModel):
    question: str
    top_k:    int = 3

class QueryResponse(BaseModel):
    question: str
    answer:   str
    sources:  list[str]
    scores:   list[float]
    context:  str


# --- App lifecycle ---

rag: RAGPipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    rag = RAGPipeline()
    yield

app = FastAPI(
    title="TorchRAG API",
    description="RAG pipeline powered by PyTorch and HuggingFace",
    version="0.1.0",
    lifespan=lifespan,
)


# --- Endpoints ---

@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/ingest")
def ingest(req: IngestRequest):
    docs = [{"title": d.title, "content": d.content} for d in req.documents]
    rag.ingest(docs)
    return {"status": "ok", "documents_ingested": len(docs)}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if rag.store.embeddings is None:
        raise HTTPException(status_code=400, detail="No documents ingested yet. Call /ingest first.")
    return rag.query(req.question, top_k=req.top_k)


@app.post("/save")
def save(path: str = "vectorstore/store.npz"):
    rag.store.save(path)
    return {"status": "ok", "path": path}


@app.post("/load")
def load(path: str = "vectorstore/store.npz"):
    rag.store.load(path)
    return {"status": "ok", "documents_loaded": len(rag.store.documents)}
