"""
AI Root Cause Analyzer - Main FastAPI Application
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

from app.analyzer import RootCauseAnalyzer
from app.ingestion import LogIngestionPipeline
from app.rag import IncidentKnowledgeBase

app = FastAPI(
    title="AI Root Cause Analyzer",
    description="Advanced DevOps tool to identify why a system failed and suggest fixes",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = RootCauseAnalyzer()
ingestion = LogIngestionPipeline()
knowledge_base = IncidentKnowledgeBase()


class AnalyzeRequest(BaseModel):
    logs: str
    metrics: Optional[str] = None
    service_name: Optional[str] = "unknown-service"
    time_window_minutes: Optional[int] = 5


class AnalyzeResponse(BaseModel):
    root_cause: str
    confidence: int
    severity: str
    evidence: list
    suggested_actions: list
    similar_past_incidents: list
    mttr_estimate_minutes: Optional[int]


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("app/static/index.html") as f:
        return f.read()


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_logs(request: AnalyzeRequest):
    """
    Main endpoint: ingest logs + metrics, identify root cause, suggest fixes.
    """
    try:
        # Step 1: Ingest and pre-process logs
        processed_logs = ingestion.process(
            raw_logs=request.logs,
            service_name=request.service_name
        )

        # Step 2: Retrieve similar past incidents from vector DB
        past_incidents = knowledge_base.search(processed_logs.summary)

        # Step 3: Run LLM reasoning chain
        result = analyzer.analyze(
            processed_logs=processed_logs,
            past_incidents=past_incidents,
            metrics=request.metrics
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest-incident")
async def ingest_incident(title: str, description: str, resolution: str):
    """
    Add a past incident (post-mortem) to the knowledge base for RAG.
    """
    knowledge_base.add_incident(title, description, resolution)
    return {"status": "ingested", "title": title}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
