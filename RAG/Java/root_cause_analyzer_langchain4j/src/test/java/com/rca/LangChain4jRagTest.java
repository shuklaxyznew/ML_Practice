package com.rca;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

/**
 * Tests the LangChain4j RAG pipeline directly:
 *   EmbeddingModel → InMemoryEmbeddingStore → semantic search
 *
 * No LLM API key required — only the local ONNX embedding model is used.
 */
class LangChain4jRagTest {

    private InMemoryEmbeddingStore<TextSegment> store;
    private EmbeddingModel embeddingModel;

    @BeforeEach
    void setUp() {
        embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        store = new InMemoryEmbeddingStore<>();

        // Ingest seed docs into the embedding store
        EmbeddingStoreIngestor ingestor = EmbeddingStoreIngestor.builder()
            .embeddingModel(embeddingModel)
            .embeddingStore(store)
            .build();

        ingestor.ingest(Document.from(
            "Title: DB connection pool exhaustion\nDescription: HikariPool maxed out. 503 errors.\nResolution: Increase pool size, kill blocking queries.",
            Metadata.from("title", "DB pool exhaustion")));

        ingestor.ingest(Document.from(
            "Title: OOM crash\nDescription: Java heap exhausted. OOMKilled.\nResolution: Increase -Xmx, paginate ETL jobs.",
            Metadata.from("title", "OOM crash")));

        ingestor.ingest(Document.from(
            "Title: Payment timeout cascade\nDescription: Circuit breaker opened. Checkout SLO breached.\nResolution: Enable fallback processor.",
            Metadata.from("title", "Payment timeout")));
    }

    @Test
    void embeddingStore_returnsResults() {
        var queryEmbedding = embeddingModel.embed("database connection pool timeout").content();
        EmbeddingSearchResult<TextSegment> result = store.search(
            EmbeddingSearchRequest.builder()
                .queryEmbedding(queryEmbedding)
                .maxResults(3)
                .minScore(0.0)
                .build()
        );
        assertThat(result.matches()).isNotEmpty();
    }

    @Test
    void semanticSearch_dbQuery_findsDbIncident() {
        var queryEmbedding = embeddingModel.embed("HikariPool connection not available jdbc").content();
        EmbeddingSearchResult<TextSegment> result = store.search(
            EmbeddingSearchRequest.builder()
                .queryEmbedding(queryEmbedding)
                .maxResults(1)
                .minScore(0.0)
                .build()
        );
        assertThat(result.matches()).isNotEmpty();
        assertThat(result.matches().get(0).embedded().text()).containsIgnoringCase("pool");
    }

    @Test
    void semanticSearch_oomQuery_findsOomIncident() {
        var queryEmbedding = embeddingModel.embed("OutOfMemoryError heap java killed").content();
        EmbeddingSearchResult<TextSegment> result = store.search(
            EmbeddingSearchRequest.builder()
                .queryEmbedding(queryEmbedding)
                .maxResults(1)
                .minScore(0.0)
                .build()
        );
        assertThat(result.matches()).isNotEmpty();
        assertThat(result.matches().get(0).embedded().text()).containsIgnoringCase("OOM");
    }

    @Test
    void similarityScores_areBetween0and1() {
        var queryEmbedding = embeddingModel.embed("payment circuit breaker timeout").content();
        EmbeddingSearchResult<TextSegment> result = store.search(
            EmbeddingSearchRequest.builder()
                .queryEmbedding(queryEmbedding)
                .maxResults(3)
                .minScore(0.0)
                .build()
        );
        result.matches().forEach(m ->
            assertThat(m.score()).isBetween(0.0, 1.0)
        );
    }

    @Test
    void newlyIngestedIncident_isImmediatelyRetrievable() {
        EmbeddingStoreIngestor.builder()
            .embeddingModel(embeddingModel)
            .embeddingStore(store)
            .build()
            .ingest(Document.from(
                "Title: Kafka consumer lag\nDescription: Consumer group fell behind 500k messages.\nResolution: Increase thread pool, reduce batch size.",
                Metadata.from("title", "Kafka lag")));

        var queryEmbedding = embeddingModel.embed("kafka consumer lag messages").content();
        EmbeddingSearchResult<TextSegment> result = store.search(
            EmbeddingSearchRequest.builder()
                .queryEmbedding(queryEmbedding)
                .maxResults(1)
                .minScore(0.0)
                .build()
        );
        assertThat(result.matches().get(0).embedded().text()).containsIgnoringCase("Kafka");
    }
}
