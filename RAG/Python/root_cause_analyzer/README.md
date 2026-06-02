# AI Root Cause Analyzer 🐞

An advanced DevOps tool that simulates a Senior SRE's troubleshooting process. Instead of just alerting on failures, it ingests logs, metrics, and error traces to identify **why** a system failed and suggests fixes based on historical incident data.

## Features

- **Log Ingestion Pipeline** — filters noise, masks PII, extracts ERROR/CRITICAL/WARN lines
- **RAG Knowledge Base** — ChromaDB vector store of past post-mortems for grounded suggestions
- **LLM Reasoner** — Claude (Anthropic) or GPT-4o via LangChain with a Senior SRE system prompt
- **REST API** — FastAPI with `/analyze` endpoint returning structured JSON
- **Confidence scoring** — tells you how sure the AI is, with evidence citations
- **MTTR estimates** — estimated time to resolve based on suggested actions

## Architecture

```
System Logs & Metrics
        │
        ▼
Data Ingestion Layer  ──► PII Masking ──► Log Summarization
        │
        ▼
    Analyzer Engine
   ┌────────────┐
   │ Log Pattern│
   │ Extraction │
   └─────┬──────┘
         │
   ┌─────▼──────┐    ┌──────────────────┐
   │ LLM        │◄───│ Past Incidents    │
   │ Reasoner   │    │ (ChromaDB / RAG)  │
   └─────┬──────┘    └──────────────────┘
         │
   ┌─────▼──────────────────┐
   │ Root Cause + Actions   │
   │ + Confidence Score     │
   └────────────────────────┘
```

## Quick Start

### 1. Clone & install

```bash
git clone <your-repo>
cd root_cause_analyzer
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Run the server

```bash
python -m app.main
# Server starts at http://localhost:8000
```

### 4. Analyze logs

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "logs": "ERROR 2026-05-06 12:01:23 [api] HikariPool-1 Connection not available timeout 30000ms\nCRITICAL Health check FAILED: DataSource failure",
    "service_name": "api-service"
  }'
```

**Response:**
```json
{
  "root_cause": "Database connection pool exhausted — all 20 connections are in use, likely due to slow queries or connection leaks",
  "confidence": 88,
  "severity": "critical",
  "evidence": [
    {"text": "HikariPool-1 Connection is not available, timeout 30000ms", "type": "error"},
    {"text": "Health check FAILED: DataSource failure", "type": "error"}
  ],
  "suggested_actions": [
    {"step": 1, "priority": "immediate", "title": "Identify blocking queries", "detail": "Run SELECT * FROM pg_stat_activity WHERE state = 'idle in transaction' and kill long-running queries."},
    {"step": 2, "priority": "short-term", "title": "Increase connection pool size", "detail": "Set maximumPoolSize=50 in HikariCP config and redeploy."},
    {"step": 3, "priority": "long-term", "title": "Add connection pool monitoring", "detail": "Expose HikariCP metrics to Prometheus and alert when pool utilization > 80%."}
  ],
  "similar_past_incidents": [...],
  "mttr_estimate_minutes": 20
}
```

## Project Structure

```
root_cause_analyzer/
├── app/
│   ├── main.py          # FastAPI app & routes
│   ├── analyzer.py      # LLM reasoning engine (LangChain)
│   ├── ingestion.py     # Log ingestion & PII masking
│   └── rag.py           # ChromaDB vector store for past incidents
├── prompts/
│   └── system_prompt.py # Senior SRE system prompt
├── data/
│   └── seed_incidents.json  # Built-in past incident knowledge base
├── tests/
│   └── test_analyzer.py     # Unit + integration tests
├── requirements.txt
├── .env.example
└── README.md
```

## Adding Past Incidents (RAG)

Add your own post-mortems or JIRA tickets to improve suggestions:

```bash
curl -X POST "http://localhost:8000/ingest-incident" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Redis OOM — 2026-03-15",
    "description": "Redis maxmemory hit during traffic spike. All writes failed.",
    "resolution": "Increased Redis memory to 16GB, enabled volatile-lru eviction policy."
  }'
```

Or edit `data/seed_incidents.json` directly and restart.

## Running Tests

```bash
pytest tests/ -v
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API Framework | FastAPI |
| LLM | Anthropic Claude / OpenAI GPT-4o |
| Orchestration | LangChain |
| Vector DB | ChromaDB (local) / Pinecone (prod) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Log Store | Elasticsearch (optional) |
| Metrics | Prometheus (optional) |

## Future Enhancements

- **Auto-Healing** — integrate with GitHub Actions / AWS Lambda for automatic rollback when confidence > 90%
- **Multi-Service Tracing** — ingest OpenTelemetry traces across microservices
- **Slack / PagerDuty Alerts** — push analysis reports to on-call channels automatically
- **Dashboard UI** — real-time incident timeline with confidence trends
