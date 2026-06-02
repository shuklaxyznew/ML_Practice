package com.rca.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.rca.model.AnalyzeResponse;
import com.rca.model.ProcessedLogs;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

/**
 * Calls the LangChain4j SreAssistant (which internally runs RAG + LLM)
 * and parses the structured JSON result into an AnalyzeResponse.
 *
 * Flow:
 *   ProcessedLogs  ──►  SreAssistant.analyze()
 *                           │
 *                           ├── RetrievalAugmentor retrieves top-3 past incidents
 *                           │   from EmbeddingStore (semantic similarity search)
 *                           │
 *                           └── LLM (Claude / GPT-4o) reasons over logs + context
 *                                   │
 *                                   └── Returns JSON  ──►  AnalyzeResponse
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class AnalyzerService {

    private final SreAssistant sreAssistant;
    private final ObjectMapper objectMapper = new ObjectMapper();

    public AnalyzeResponse analyze(ProcessedLogs logs, String metrics) {
        String metricsSection = (metrics != null && !metrics.isBlank())
            ? "=== METRICS ===\n" + metrics
            : "";

        log.info("Invoking LangChain4j SreAssistant for service: {}", logs.getServiceName());

        // LangChain4j automatically:
        //  1. Embeds the prompt
        //  2. Retrieves relevant past incidents from EmbeddingStore (RAG)
        //  3. Injects them into the prompt
        //  4. Calls the configured ChatLanguageModel
        String rawJson = sreAssistant.analyze(
            logs.getServiceName(),
            logs.getErrorCount(),
            logs.getWarningCount(),
            logs.getSummary(),
            metricsSection
        );

        log.debug("LLM raw response:\n{}", rawJson);
        return parseResponse(rawJson);
    }

    private AnalyzeResponse parseResponse(String raw) {
        try {
            // Strip markdown fences if the LLM adds them despite instructions
            String clean = raw.strip();
            if (clean.startsWith("```")) {
                clean = clean.lines().skip(1)
                    .takeWhile(l -> !l.equals("```"))
                    .reduce("", (a, b) -> a + "\n" + b).strip();
            }

            JsonNode node = objectMapper.readTree(clean);

            List<AnalyzeResponse.Evidence> evidence = new ArrayList<>();
            node.path("evidence").forEach(e -> evidence.add(
                AnalyzeResponse.Evidence.builder()
                    .text(e.path("text").asText())
                    .type(e.path("type").asText("error"))
                    .build()
            ));

            List<AnalyzeResponse.ActionItem> actions = new ArrayList<>();
            node.path("suggested_actions").forEach(a -> actions.add(
                AnalyzeResponse.ActionItem.builder()
                    .step(a.path("step").asInt())
                    .priority(a.path("priority").asText("short-term"))
                    .title(a.path("title").asText())
                    .detail(a.path("detail").asText())
                    .build()
            ));

            List<String> services = new ArrayList<>();
            node.path("affected_services").forEach(s -> services.add(s.asText()));

            return AnalyzeResponse.builder()
                .rootCause(node.path("root_cause").asText())
                .confidence(node.path("confidence").asInt(50))
                .severity(node.path("severity").asText("medium"))
                .errorCount(node.path("error_count").asInt())
                .warningCount(node.path("warning_count").asInt())
                .affectedServices(services)
                .mttrEstimateMinutes(node.hasNonNull("mttr_estimate_minutes")
                    ? node.path("mttr_estimate_minutes").asInt() : null)
                .evidence(evidence)
                .suggestedActions(actions)
                .build();

        } catch (Exception e) {
            throw new RuntimeException(
                "Failed to parse LLM response as JSON: " + e.getMessage() + "\nRaw:\n" + raw, e);
        }
    }
}
