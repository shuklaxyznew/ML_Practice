# AI Root Cause Analyzer 🐞 — Java / Spring Boot

Java port of the AI Root Cause Analyzer. Built with **Spring Boot 3**, **OkHttp**, and **Anthropic Claude** (or OpenAI GPT-4o).

## Tech Stack

| Component       | Technology                            |
|-----------------|---------------------------------------|
| Framework       | Spring Boot 3.3 (Java 21)             |
| REST API        | Spring MVC + Swagger UI               |
| LLM             | Anthropic Claude via OkHttp (default) |
| LLM alternative | OpenAI GPT-4o                         |
| RAG / Vector DB | In-memory keyword search (built-in)   |
| Production RAG  | Swap for Spring AI + PGVector/Pinecone|
| Build           | Maven                                 |

## Project Structure

```
src/
├── main/java/com/rca/
│   ├── RootCauseAnalyzerApplication.java   # Spring Boot entry point
│   ├── controller/
│   │   └── AnalyzerController.java         # REST endpoints
│   ├── service/
│   │   ├── RootCauseOrchestrator.java      # Pipeline orchestrator
│   │   └── AnalyzerService.java            # LLM call + response parsing
│   ├── ingestion/
│   │   └── LogIngestionPipeline.java       # PII masking, log filtering
│   ├── rag/
│   │   └── IncidentKnowledgeBase.java      # Past incident store + search
│   ├── prompt/
│   │   └── SystemPrompt.java               # Senior SRE system prompt
│   └── model/
│       ├── AnalyzeRequest.java
│       ├── AnalyzeResponse.java
│       ├── ProcessedLogs.java
│       └── Incident.java
├── main/resources/
│   ├── application.properties
│   └── seed_incidents.json                 # Built-in incident knowledge base
└── test/java/com/rca/
    ├── LogIngestionPipelineTest.java
    └── IncidentKnowledgeBaseTest.java
```

## Quick Start

### 1. Prerequisites

- Java 21+
- Maven 3.8+

### 2. Build

```bash
mvn clean package -DskipTests
```

### 3. Configure API key

```bash
export ANTHROPIC_API_KEY=your_key_here
# or for OpenAI:
# export OPENAI_API_KEY=your_key_here
```

Or edit `src/main/resources/application.properties`:
```properties
anthropic.api.key=your_key_here
```

### 4. Run

```bash
java -jar target/root-cause-analyzer-1.0.0.jar
# Starts at http://localhost:8080
```

### 5. Open Swagger UI

```
http://localhost:8080/swagger-ui
```

## API Usage

### POST /api/v1/analyze

```bash
curl -X POST http://localhost:8080/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "logs": "ERROR 2026-05-06 12:01:23 [api] HikariPool-1 Connection is not available, timeout 30000ms\nCRITICAL 12:01:25 [api] Health check FAILED: DataSource failure\nWARN 12:00:58 [api] Pool usage 95%: 19/20 connections active",
    "serviceName": "api-service"
  }'
```

**Response:**
```json
{
  "rootCause": "Database connection pool exhausted — all connections in use due to slow/blocking queries",
  "confidence": 88,
  "severity": "critical",
  "errorCount": 2,
  "warningCount": 1,
  "affectedServices": ["api-service"],
  "mttrEstimateMinutes": 20,
  "evidence": [
    { "text": "HikariPool-1 Connection is not available, timeout 30000ms", "type": "error" },
    { "text": "Health check FAILED: DataSource failure", "type": "error" }
  ],
  "suggestedActions": [
    {
      "step": 1, "priority": "immediate",
      "title": "Kill blocking DB queries",
      "detail": "Run SELECT * FROM pg_stat_activity WHERE state = 'idle in transaction' and terminate long-running queries."
    },
    {
      "step": 2, "priority": "short-term",
      "title": "Increase HikariCP pool size",
      "detail": "Set maximumPoolSize=50 and connectionTimeout=10000 in HikariCP config."
    }
  ],
  "similarPastIncidents": [
    {
      "title": "DB connection pool exhaustion — 2025-11-03",
      "resolution": "Increased pool size to 50, added query timeout of 10s, killed blocking queries.",
      "similarity": 0.9
    }
  ]
}
```

### POST /api/v1/incidents — Add to knowledge base

```bash
curl -X POST "http://localhost:8080/api/v1/incidents" \
  -d "title=Kafka lag spike" \
  -d "description=Consumer group fell behind due to slow message processing" \
  -d "resolution=Increased consumer thread pool, reduced batch size"
```

## Running Tests

```bash
mvn test
```

## Upgrading to Production RAG

Replace `IncidentKnowledgeBase` keyword search with semantic vector search:

```xml
<!-- Add to pom.xml -->
<dependency>
    <groupId>org.springframework.ai</groupId>
    <artifactId>spring-ai-pgvector-store-spring-boot-starter</artifactId>
    <version>1.0.0</version>
</dependency>
```

Then wire `VectorStore` from Spring AI — the rest of the code stays the same.

## Future Enhancements

- **Auto-Healing** — trigger GitHub Actions / AWS Lambda on confidence > 90%
- **Slack/PagerDuty integration** — push analysis reports to on-call channels
- **OpenTelemetry tracing** — ingest distributed traces across microservices
- **Prometheus metrics** — expose MTTR trends, confidence histograms
