package com.rca.config;

import com.rca.service.SreAssistant;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.anthropic.AnthropicChatModel;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.List;

/**
 * LangChain4j wiring:
 *
 *  Python LangChain equivalent mapping:
 *  ─────────────────────────────────────────────────────────────────────
 *  LangChain (Python)                  LangChain4j (Java)
 *  ─────────────────────────────────────────────────────────────────────
 *  ChatAnthropic / ChatOpenAI          AnthropicChatModel / OpenAiChatModel
 *  RetrievalQA chain                   AiServices + RetrievalAugmentor
 *  VectorStore (Chroma/Pinecone)       EmbeddingStore (InMemory / PGVector)
 *  SentenceTransformerEmbeddings       AllMiniLmL6V2EmbeddingModel (ONNX)
 *  PromptTemplate + chain.invoke()     @SystemMessage + @UserMessage on interface
 *  ConversationBufferMemory            MessageWindowChatMemory
 *  ─────────────────────────────────────────────────────────────────────
 */
@Slf4j
@Configuration
public class LangChain4jConfig {

    @Value("${anthropic.api.key:}")
    private String anthropicKey;

    @Value("${openai.api.key:}")
    private String openAiKey;

    // ── 1. Chat Language Model (Claude default, OpenAI fallback) ──────────────

    @Bean
    public ChatLanguageModel chatLanguageModel() {
        if (anthropicKey != null && !anthropicKey.isBlank()) {
            log.info("LangChain4j: Using Anthropic Claude (claude-sonnet-4-20250514)");
            return AnthropicChatModel.builder()
                .apiKey(anthropicKey)
                .modelName("claude-sonnet-4-20250514")
                .maxTokens(1500)
                .temperature(0.0)
                .build();
        }
        if (openAiKey != null && !openAiKey.isBlank()) {
            log.info("LangChain4j: Using OpenAI GPT-4o");
            return OpenAiChatModel.builder()
                .apiKey(openAiKey)
                .modelName("gpt-4o")
                .maxTokens(1500)
                .temperature(0.0)
                .build();
        }
        throw new IllegalStateException(
            "No LLM key configured. Set anthropic.api.key or openai.api.key in application.properties"
        );
    }

    // ── 2. Embedding Model — local ONNX (all-MiniLM-L6-v2) ──────────────────
    //    Same model as Python's sentence-transformers, runs fully offline.

    @Bean
    public EmbeddingModel embeddingModel() {
        return new AllMiniLmL6V2EmbeddingModel();
    }

    // ── 3. Embedding Store — in-memory (swap for PGVector/Pinecone in prod) ──

    @Bean
    public EmbeddingStore<TextSegment> embeddingStore(
            EmbeddingModel embeddingModel,
            List<Document> seedIncidentDocuments) {

        InMemoryEmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();

        // Ingest seed incidents into the vector store at startup
        EmbeddingStoreIngestor ingestor = EmbeddingStoreIngestor.builder()
            .embeddingModel(embeddingModel)
            .embeddingStore(store)
            .build();

        ingestor.ingest(seedIncidentDocuments);
        log.info("LangChain4j: Ingested {} seed incidents into EmbeddingStore", seedIncidentDocuments.size());

        return store;
    }

    // ── 4. RAG — RetrievalAugmentor (equivalent to LangChain's RetrievalQA) ──

    @Bean
    public RetrievalAugmentor retrievalAugmentor(
            EmbeddingStore<TextSegment> embeddingStore,
            EmbeddingModel embeddingModel) {

        EmbeddingStoreContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
            .embeddingStore(embeddingStore)
            .embeddingModel(embeddingModel)
            .maxResults(3)            // top-3 most similar past incidents
            .minScore(0.4)            // minimum cosine similarity threshold
            .build();

        return DefaultRetrievalAugmentor.builder()
            .contentRetriever(contentRetriever)
            .build();
    }

    // ── 5. AI Service — declarative LLM interface with RAG auto-injected ─────

    @Bean
    public SreAssistant sreAssistant(
            ChatLanguageModel chatLanguageModel,
            RetrievalAugmentor retrievalAugmentor) {

        return AiServices.builder(SreAssistant.class)
            .chatLanguageModel(chatLanguageModel)
            .retrievalAugmentor(retrievalAugmentor)   // RAG wired here
            .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
            .build();
    }

    // ── 6. Seed Documents — past incidents loaded as LangChain4j Documents ───

    @Bean
    public List<Document> seedIncidentDocuments() {
        return List.of(
            incident("DB connection pool exhaustion — 2025-11-03",
                "HikariPool max connections reached. API returned 503. Long-running queries blocked connections.",
                "Increased pool size to 50, added 10s query timeout, killed blocking queries via pg_stat_activity."),

            incident("OOM crash — worker nodes — 2025-09-14",
                "Worker pods killed with OOMKilled. Java heap usage climbed to 100% during bulk ETL job.",
                "Increased JVM -Xmx to 4G, paginated ETL job, set K8s memory limit to 5Gi."),

            incident("Payment service timeout cascade — 2025-08-22",
                "Downstream payment provider had elevated latency. Circuit breaker opened. Checkout success rate dropped to 34%.",
                "Enabled fallback payment processor, tuned circuit breaker thresholds, added retry with exponential backoff."),

            incident("K8s CrashLoopBackOff — missing env var — 2025-07-10",
                "Deployment without required DATABASE_URL env var. All pods exited code 1 on startup.",
                "Added env var to K8s secret, re-deployed. Added startup validation to catch missing config early."),

            incident("Redis cache eviction storm — 2025-06-05",
                "Redis maxmemory reached. allkeys-lru eviction kicked in. Cache hit rate dropped from 95% to 12%, DB overloaded.",
                "Increased Redis memory to 8GB, switched to volatile-lru eviction, added cache warming on startup."),

            incident("Elasticsearch disk I/O saturation — 2025-05-18",
                "ES nodes hit disk I/O saturation during large index operation. Search latency spiked 10x.",
                "Throttled bulk indexing with refresh_interval=30s, added dedicated data nodes, migrated to SSD.")
        );
    }

    private Document incident(String title, String description, String resolution) {
        String text = String.format("Title: %s\nDescription: %s\nResolution: %s",
            title, description, resolution);
        return Document.from(text, Metadata.from("title", title));
    }
}
