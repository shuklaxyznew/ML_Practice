package com.rca;

import com.rca.model.Incident;
import com.rca.rag.IncidentKnowledgeBase;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.*;

class IncidentKnowledgeBaseTest {

    private IncidentKnowledgeBase kb;

    @BeforeEach
    void setUp() {
        kb = new IncidentKnowledgeBase();
        kb.init(); // trigger @PostConstruct manually
    }

    @Test
    void search_dbKeywords_returnsDbIncident() {
        List<Incident> results = kb.search("database connection pool hikari jdbc", 3);

        assertThat(results).isNotEmpty();
        assertThat(results.get(0).getTitle()).containsIgnoringCase("pool");
    }

    @Test
    void search_oomKeywords_returnsOomIncident() {
        List<Incident> results = kb.search("OutOfMemoryError heap java killed", 3);

        assertThat(results).isNotEmpty();
    }

    @Test
    void search_noMatch_returnsEmptyList() {
        List<Incident> results = kb.search("completely unrelated query xyz123qwe", 3);

        assertThat(results).isEmpty();
    }

    @Test
    void search_returnsAtMostTopK() {
        List<Incident> results = kb.search("error memory connection timeout crash", 3);

        assertThat(results.size()).isLessThanOrEqualTo(3);
    }

    @Test
    void similarityScore_isBetween0and1() {
        List<Incident> results = kb.search("timeout circuit breaker payment", 3);

        results.forEach(r ->
            assertThat(r.getSimilarityScore()).isBetween(0.0, 1.0)
        );
    }

    @Test
    void addIncident_thenSearchFindsIt() {
        kb.addIncident(
            "Kafka consumer lag spike — 2026-01-10",
            "Consumer group fell behind due to slow message processing. Lag grew to 500k messages.",
            "Increased consumer thread pool, reduced batch size, added lag alerting."
        );

        List<Incident> results = kb.search("kafka consumer lag messages processing", 3);
        assertThat(results).anyMatch(i -> i.getTitle().contains("Kafka"));
    }

    @Test
    void resultsSortedByDescendingSimilarity() {
        List<Incident> results = kb.search("database pool connection timeout", 5);

        for (int i = 0; i < results.size() - 1; i++) {
            assertThat(results.get(i).getSimilarityScore())
                .isGreaterThanOrEqualTo(results.get(i + 1).getSimilarityScore());
        }
    }
}
