# Enterprise Incident Resolution Agent

A production-grade multi-agent AI platform that autonomously investigates 
incidents, gathers information from multiple sources, reasons about root 
causes, and generates actionable recommendations.

## Architecture
Incident Input → Coordinator Agent → Knowledge Agent → Resolution Agent

↓                    ↓                  ↓

Task Planning      Runbook Search       Root Cause Analysis

Workflow Control   Log Parsing          Recommendations

Incident Lookup    Vector Retrieval     Confidence Scoring

↓                    ↓                  ↓

LangGraph StateGraph → Reflection Node → Structured Report

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM Runtime | Ollama (local, CPU) |
| Planning Model | Qwen 2.5 3B |
| Reasoning Model | Gemma 3 4B |
| Orchestration | LangGraph |
| Agent Collaboration | CrewAI |
| Framework | LangChain |
| Vector Store | ChromaDB |
| Embeddings | BGE-Small-EN-v1.5 |
| Search | DuckDuckGo |
| API Layer | FastAPI |
| MCP Server | FastMCP |

## Features

- **Multi-Agent Orchestration** — LangGraph StateGraph with 3 specialized agents
- **ReAct Reasoning** — Thought → Action → Observation loops per agent
- **Reflection & Retry** — Confidence-based self-evaluation with automatic retry
- **5 Custom Tools** — Search, log parser, incident lookup, vector retrieval, similarity
- **FastAPI Webhook** — External systems trigger investigations via REST API
- **MCP Server** — 6 enterprise tools exposed via Model Context Protocol
- **Evaluation Pipeline** — Automated quality scoring across 5 dimensions
- **Session + Historical Memory** — In-process + SQLite persistence

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/incident-agent
cd incident-agent
py -3.12 -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Pull models
ollama pull qwen2.5:3b
ollama pull gemma3:4b

# 4. Seed knowledge base
python -m knowledge_base.ingest

# 5. Run a single investigation
python main.py

# 6. Start the API server
uvicorn api:app --host 0.0.0.0 --port 8080 --reload
```

## API Usage

```bash
# Submit an incident
curl -X POST http://localhost:8080/incidents \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Payment service returning 500 errors",
    "description": "Error rate jumped to 45% at 14:32 UTC",
    "affected_service": "payment-service",
    "severity": "P1",
    "raw_logs": "ERROR: Connection pool exhausted 50/50"
  }'

# Poll for results
curl http://localhost:8080/incidents/{job_id}

# Health check
curl http://localhost:8080/health
```

## Agent Workflow

intake_node       — validates incident, sets initial state
coordinator_node  — plans investigation, looks up past incidents
knowledge_node    — searches runbooks, parses logs, finds similar incidents
resolution_node   — generates RCA + recommendations (Gemma 3 4B)
reflection_node   — scores confidence, retries if below threshold
output_node       — assembles and formats final report


## Testing

```bash
# Unit + integration tests (no LLM required)
python -m pytest tests/unit/ tests/integration/ -v

# Full E2E test (requires Ollama)
python -m pytest tests/e2e/ -v -s
```

## Project Structure
incident-agent/

├── agents/          # Agent definitions (coordinator, knowledge, resolution)

├── workflows/       # LangGraph state, graph, router

├── tools/           # 5 custom LangChain tools

├── memory/          # Session + historical memory

├── knowledge_base/  # ChromaDB ingestion + retrieval

├── crews/           # CrewAI crew + task definitions

├── mcp/             # FastMCP server with 6 tools

├── evaluation/      # Metrics + automated evaluation

├── observability/   # Logging, cost tracking, error handling

├── config/          # Pydantic settings

├── api.py           # FastAPI webhook layer

├── main.py          # CLI entry point

└── tests/           # Unit, integration, E2E

## Hardware Requirements

Runs entirely on CPU. Tested on Intel i5-4440, 12GB RAM.

- Qwen 2.5 3B: ~2GB RAM, ~8-15 tokens/sec
- Gemma 3 4B: ~3GB RAM, ~6-10 tokens/sec  
- ChromaDB + BGE-Small: ~500MB RAM
- Peak usage (sequential): ~4GB RAM
