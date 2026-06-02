package com.rca.rag;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.rca.model.Incident;
import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

/**
 * In-memory incident knowledge base for RAG.
 *
 * In production, swap this for:
 *   - Pinecone / Weaviate / Qdrant (cloud vector DB)
 *   - Spring AI VectorStore with PGVector (Postgres extension)
 *
 * Similarity is computed via TF-IDF-style keyword overlap.
 * For semantic search, integrate sentence-transformers via DJL or a REST embedding endpoint.
 */
@Slf4j
@Service
public class IncidentKnowledgeBase {

    private final List<Incident> incidents = new ArrayList<>();
    private final ObjectMapper objectMapper = new ObjectMapper();

    @PostConstruct
    public void init() {
        loadSeedIncidents();
        log.info("Knowledge base initialized with {} incidents", incidents.size());
    }

    public List<Incident> search(String query, int topK) {
        String normalizedQuery = query.toLowerCase();
        String[] queryTokens = normalizedQuery.split("\\W+");

        return incidents.stream()
            .map(inc -> {
                double score = computeSimilarity(inc, queryTokens, normalizedQuery);
                return Incident.builder()
                    .title(inc.getTitle())
                    .description(inc.getDescription())
                    .resolution(inc.getResolution())
                    .keywords(inc.getKeywords())
                    .similarityScore(score)
                    .build();
            })
            .filter(inc -> inc.getSimilarityScore() > 0)
            .sorted(Comparator.comparingDouble(Incident::getSimilarityScore).reversed())
            .limit(topK)
            .collect(Collectors.toList());
    }

    public void addIncident(String title, String description, String resolution) {
        List<String> keywords = Arrays.stream(
            (description + " " + resolution).toLowerCase().split("\\W+"))
            .filter(t -> t.length() > 3)
            .distinct()
            .collect(Collectors.toList());

        incidents.add(Incident.builder()
            .title(title)
            .description(description)
            .resolution(resolution)
            .keywords(keywords)
            .build());
        log.info("Ingested new incident: {}", title);
    }

    private double computeSimilarity(Incident incident, String[] queryTokens, String query) {
        if (incident.getKeywords() == null) return 0;

        // Keyword hit count
        long hits = incident.getKeywords().stream()
            .filter(query::contains)
            .count();

        // Bonus: description/resolution match
        String body = (incident.getDescription() + " " + incident.getResolution()).toLowerCase();
        long bodyHits = Arrays.stream(queryTokens)
            .filter(t -> t.length() > 3 && body.contains(t))
            .count();

        double raw = hits * 0.3 + bodyHits * 0.1;
        return Math.min(1.0, raw);
    }

    private void loadSeedIncidents() {
        try {
            ClassPathResource resource = new ClassPathResource("seed_incidents.json");
            List<Incident> seed = objectMapper.readValue(
                resource.getInputStream(),
                new TypeReference<>() {}
            );
            incidents.addAll(seed);
        } catch (IOException e) {
            log.warn("Could not load seed incidents, using built-in defaults: {}", e.getMessage());
            incidents.addAll(builtinIncidents());
        }
    }

    private List<Incident> builtinIncidents() {
        return List.of(
            Incident.builder()
                .title("DB connection pool exhaustion — 2025-11-03")
                .description("HikariPool max connections reached. API returned 503. Long-running queries blocked connections.")
                .resolution("Increased pool size to 50, added 10s query timeout, killed blocking queries.")
                .keywords(List.of("hikari","connection","pool","jdbc","database","503","timeout"))
                .build(),
            Incident.builder()
                .title("OOM crash — worker nodes — 2025-09-14")
                .description("Worker pods killed with OOMKilled. Heap usage climbed to 100% during bulk ETL job.")
                .resolution("Increased JVM -Xmx to 4G, paginated ETL job, set K8s memory limit to 5Gi.")
                .keywords(List.of("oom","outofmemory","heap","gc","memory","killed","oomed","java"))
                .build(),
            Incident.builder()
                .title("Payment service timeout cascade — 2025-08-22")
                .description("Downstream payment provider elevated latency. Circuit breaker opened. Checkout success dropped.")
                .resolution("Enabled fallback processor, tuned circuit breaker thresholds, added retry with jitter.")
                .keywords(List.of("timeout","circuit","breaker","payment","latency","cascade","retry","503"))
                .build(),
            Incident.builder()
                .title("K8s CrashLoopBackOff — missing env var — 2025-07-10")
                .description("Deployment without required DATABASE_URL. All pods exited code 1 on startup.")
                .resolution("Added env var to K8s secret, re-deployed. Added startup validation to catch missing config.")
                .keywords(List.of("crashloop","crash","env","environment","variable","startup","exit","kubernetes"))
                .build()
        );
    }
}
