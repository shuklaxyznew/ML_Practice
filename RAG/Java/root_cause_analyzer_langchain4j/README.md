# AI Root Cause Analyzer 🐞 — Java + LangChain4j

Java implementation of the AI Root Cause Analyzer using **LangChain4j** — the Java equivalent of LangChain (Python).

## LangChain Python → LangChain4j Java Mapping

| LangChain (Python)                   | LangChain4j (Java)                          |
|--------------------------------------|---------------------------------------------|
| `ChatAnthropic` / `ChatOpenAI`       | `AnthropicChatModel` / `OpenAiChatModel`    |
| `RetrievalQA` chain                  | `AiServices` + `RetrievalAugmentor`         |
| `Chroma` / `Pinecone` VectorStore    | `InMemoryEmbeddingStore` / `PgVectorEmbeddingStore` |
| `sentence-transformers` embeddings   | `AllMiniLmL6V2EmbeddingModel` (ONNX, local) |
| `PromptTemplate` + `chain.invoke()`  | `@SystemMessage` + `@UserMessage` on interface |
| `ConversationBufferMemory`           | `MessageWindowChatMemory`                   |
| `VectorStore.add_documents()`        | `EmbeddingStoreIngestor.ingest()`           |

## Architecture

```
AnalyzeRequest (logs + metrics)
        │
        ▼
LogIngestionPipeline          ← PII masking, error/warn filtering, summary (max 30 lines)
        │
        ▼
AnalyzerService
  └── SreAssistant.analyze()  ← LangChain4j @AiService interface
          │
          ├── RetrievalAugmentor  ← auto-retrieves top-3 past incidents from EmbeddingStore (RAG)
          │       └── AllMiniLmL6V2 embeddings + InMemoryEmbeddingStore
          │
          └── ChatLanguageModel  ← Anthropic Claude (default) or OpenAI GPT-4o
                  │
                  └── JSON response  →  AnalyzeResponse
```

## Project Structure

```
src/
├── main/java/com/rca/
│   ├── RootCauseAnalyzerApplication.java
│   ├── config/
│   │   └── LangChain4jConfig.java          # Wires ChatModel, EmbeddingModel, EmbeddingStore, RAG
│   ├── service/
│   │   ├── SreAssistant.java               # LangChain4j @AiService interface (= LangChain chain)
│   │   ├── AnalyzerService.java            # Invokes SreAssistant, parses JSON
│   │   └── RootCauseOrchestrator.java      # Pipeline coordinator
│   ├── ingestion/
│   │   └── LogIngestionPipeline.java       # PII masking, filtering, summarization
│   ├── rag/
│   │   └── IncidentKnowledgeBase.java      # Runtime incident ingestion into EmbeddingStore
│   ├── controller/
│   │   └── AnalyzerController.java         # REST: POST /api/v1/analyze
│   └── model/                              # POJOs: AnalyzeRequest, AnalyzeResponse, ProcessedLogs
├── main/resources/
│   └── application.properties
└── test/java/com/rca/
    ├── LogIngestionPipelineTest.java        # Unit tests (no API key needed)
    └── LangChain4jRagTest.java             # RAG pipeline tests (no API key needed)
```

## Quick Start

### 1. Prerequisites
- Java 21+
- Maven 3.8+

### 2. Build

```bash
mvn clean package -DskipTests
```

### 3. Configure API Key

```bash
export ANTHROPIC_API_KEY=your_anthropic_key_here
# or: export OPENAI_API_KEY=your_openai_key_here
```

### 4. Run

```bash
java -jar target/root-cause-analyzer-1.0.0.jar
# → http://localhost:8080
# → Swagger UI: http://localhost:8080/swagger-ui
```

## API Usage

### POST /api/v1/analyze

```bash
curl -X POST http://localhost:8080/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "logs": "ERROR 2026-05-06 [api] HikariPool-1 Connection not available timeout 30000ms\nCRITICAL [api] Health check FAILED: DataSource failure\nWARN [api] Pool usage 95%: 19/20 active",
    "serviceName": "api-service"
  }'
```

**Response:**
```json
{
  "rootCause": "Database connection pool exhausted — all connections in use, likely due to slow or blocking queries",
  "confidence": 88,
  "severity": "critical",
  "errorCount": 2,
  "warningCount": 1,
  "affectedServices": ["api-service"],
  "mttrEstimateMinutes": 20,
  "evidence": [
    { "text": "HikariPool-1 Connection not available timeout 30000ms", "type": "error" },
    { "text": "Health check FAILED: DataSource failure", "type": "error" }
  ],
  "suggestedActions": [
    { "step": 1, "priority": "immediate", "title": "Kill blocking queries",
      "detail": "Run SELECT * FROM pg_stat_activity WHERE state = 'idle in transaction' and terminate long-running sessions." },
    { "step": 2, "priority": "short-term", "title": "Increase HikariCP pool size",
      "detail": "Set maximumPoolSize=50 and connectionTimeout=10000 in your HikariCP datasource config and redeploy." },
    { "step": 3, "priority": "long-term", "title": "Add pool utilization alerting",
      "detail": "Expose HikariCP metrics to Prometheus, alert when pool utilization exceeds 80%." }
  ]
}
```

### POST /api/v1/incidents — Add incident to RAG store

```bash
curl -X POST "http://localhost:8080/api/v1/incidents" \
  -d "title=Kafka consumer lag" \
  -d "description=Consumer group fell behind 500k messages due to slow processing" \
  -d "resolution=Increased consumer thread pool, reduced batch size, added lag alerting"
```

## Running Tests

The ingestion and RAG tests run fully offline — no LLM API key needed:

```bash
mvn test
```

## Upgrading to Production Vector DB

In `LangChain4jConfig.java`, swap `InMemoryEmbeddingStore` for any of:

```xml
<!-- PGVector (Postgres) -->
<dependency>
    <groupId>dev.langchain4j</groupId>
    <artifactId>langchain4j-pgvector</artifactId>
    <version>0.35.0</version>
</dependency>

<!-- Pinecone -->
<dependency>
    <groupId>dev.langchain4j</groupId>
    <artifactId>langchain4j-pinecone</artifactId>
    <version>0.35.0</version>
</dependency>
```

Then in config:
```java
// Replace InMemoryEmbeddingStore with:
PgVectorEmbeddingStore.builder()
    .host("localhost").port(5432)
    .database("rca_db").user("postgres").password("...")
    .table("incidents").dimension(384)
    .build();
```
