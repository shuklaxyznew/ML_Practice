package com.rca.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.rca.model.AnalyzeResponse;
import com.rca.model.Incident;
import com.rca.model.ProcessedLogs;
import com.rca.prompt.SystemPrompt;
import lombok.extern.slf4j.Slf4j;
import okhttp3.*;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * LLM Reasoner — calls Anthropic Claude API directly via OkHttp.
 * Parses structured JSON response into AnalyzeResponse.
 *
 * To switch to OpenAI GPT-4o, set openai.api.key and update buildRequestBody().
 */
@Slf4j
@Service
public class AnalyzerService {

    private static final String ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages";
    private static final String ANTHROPIC_MODEL   = "claude-sonnet-4-20250514";
    private static final MediaType JSON_MEDIA      = MediaType.get("application/json");

    @Value("${anthropic.api.key:}")
    private String anthropicApiKey;

    @Value("${openai.api.key:}")
    private String openAiApiKey;

    private final ObjectMapper objectMapper = new ObjectMapper();
    private final OkHttpClient httpClient = new OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .build();

    public AnalyzeResponse analyze(ProcessedLogs processedLogs,
                                   List<Incident> pastIncidents,
                                   String metrics) {
        String ragContext  = formatRagContext(pastIncidents);
        String userPrompt  = buildUserPrompt(processedLogs, ragContext, metrics);

        String rawJson = callLlm(userPrompt);
        return parseResponse(rawJson, pastIncidents);
    }

    // ── LLM Call ──────────────────────────────────────────────────────────────

    private String callLlm(String userPrompt) {
        if (anthropicApiKey != null && !anthropicApiKey.isBlank()) {
            return callAnthropic(userPrompt);
        }
        if (openAiApiKey != null && !openAiApiKey.isBlank()) {
            return callOpenAi(userPrompt);
        }
        throw new IllegalStateException(
            "No LLM API key configured. Set anthropic.api.key or openai.api.key in application.properties"
        );
    }

    private String callAnthropic(String userPrompt) {
        try {
            ObjectNode body = objectMapper.createObjectNode();
            body.put("model", ANTHROPIC_MODEL);
            body.put("max_tokens", 1500);
            body.put("system", SystemPrompt.SRE_SYSTEM_PROMPT);

            ArrayNode messages = body.putArray("messages");
            ObjectNode msg = messages.addObject();
            msg.put("role", "user");
            msg.put("content", userPrompt);

            Request request = new Request.Builder()
                .url(ANTHROPIC_API_URL)
                .addHeader("x-api-key", anthropicApiKey)
                .addHeader("anthropic-version", "2023-06-01")
                .addHeader("Content-Type", "application/json")
                .post(RequestBody.create(objectMapper.writeValueAsString(body), JSON_MEDIA))
                .build();

            try (Response response = httpClient.newCall(request).execute()) {
                if (!response.isSuccessful()) {
                    throw new RuntimeException("Anthropic API error: " + response.code() + " " + response.message());
                }
                JsonNode resp = objectMapper.readTree(response.body().string());
                return resp.path("content").get(0).path("text").asText();
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to call Anthropic API", e);
        }
    }

    private String callOpenAi(String userPrompt) {
        try {
            ObjectNode body = objectMapper.createObjectNode();
            body.put("model", "gpt-4o");
            body.put("max_tokens", 1500);
            body.put("temperature", 0);

            ArrayNode messages = body.putArray("messages");
            ObjectNode sysMsg = messages.addObject();
            sysMsg.put("role", "system");
            sysMsg.put("content", SystemPrompt.SRE_SYSTEM_PROMPT);

            ObjectNode userMsg = messages.addObject();
            userMsg.put("role", "user");
            userMsg.put("content", userPrompt);

            Request request = new Request.Builder()
                .url("https://api.openai.com/v1/chat/completions")
                .addHeader("Authorization", "Bearer " + openAiApiKey)
                .addHeader("Content-Type", "application/json")
                .post(RequestBody.create(objectMapper.writeValueAsString(body), JSON_MEDIA))
                .build();

            try (Response response = httpClient.newCall(request).execute()) {
                if (!response.isSuccessful()) {
                    throw new RuntimeException("OpenAI API error: " + response.code());
                }
                JsonNode resp = objectMapper.readTree(response.body().string());
                return resp.path("choices").get(0).path("message").path("content").asText();
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to call OpenAI API", e);
        }
    }

    // ── Prompt Building ───────────────────────────────────────────────────────

    private String buildUserPrompt(ProcessedLogs logs, String ragContext, String metrics) {
        StringBuilder sb = new StringBuilder();
        sb.append("Service: ").append(logs.getServiceName()).append("\n");
        sb.append("Errors: ").append(logs.getErrorCount())
          .append(" | Warnings: ").append(logs.getWarningCount()).append("\n\n");
        sb.append("=== CURRENT LOGS ===\n").append(logs.getSummary());
        if (metrics != null && !metrics.isBlank()) {
            sb.append("\n\n=== METRICS ===\n").append(metrics);
        }
        sb.append("\n\n=== SIMILAR PAST INCIDENTS ===\n").append(ragContext);
        return sb.toString();
    }

    private String formatRagContext(List<Incident> incidents) {
        if (incidents == null || incidents.isEmpty()) {
            return "No similar past incidents found.";
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < incidents.size(); i++) {
            Incident inc = incidents.get(i);
            sb.append(i + 1).append(". [")
              .append(String.format("%.0f%%", inc.getSimilarityScore() * 100))
              .append(" match] ").append(inc.getTitle()).append("\n");
            sb.append("   Resolution: ").append(inc.getResolution()).append("\n");
        }
        return sb.toString();
    }

    // ── Response Parsing ──────────────────────────────────────────────────────

    private AnalyzeResponse parseResponse(String raw, List<Incident> pastIncidents) {
        try {
            String clean = raw.strip();
            if (clean.startsWith("```")) {
                clean = clean.lines().skip(1).reduce("", (a, b) -> a + "\n" + b).strip();
            }
            if (clean.endsWith("```")) {
                clean = clean.substring(0, clean.lastIndexOf("```")).strip();
            }

            JsonNode node = objectMapper.readTree(clean);

            List<AnalyzeResponse.Evidence> evidence = new ArrayList<>();
            if (node.has("evidence")) {
                for (JsonNode e : node.get("evidence")) {
                    evidence.add(AnalyzeResponse.Evidence.builder()
                        .text(e.path("text").asText())
                        .type(e.path("type").asText("error"))
                        .build());
                }
            }

            List<AnalyzeResponse.ActionItem> actions = new ArrayList<>();
            if (node.has("suggested_actions")) {
                for (JsonNode a : node.get("suggested_actions")) {
                    actions.add(AnalyzeResponse.ActionItem.builder()
                        .step(a.path("step").asInt())
                        .priority(a.path("priority").asText("short-term"))
                        .title(a.path("title").asText())
                        .detail(a.path("detail").asText())
                        .build());
                }
            }

            List<String> services = new ArrayList<>();
            if (node.has("affected_services")) {
                node.get("affected_services").forEach(s -> services.add(s.asText()));
            }

            List<AnalyzeResponse.SimilarIncident> similar = pastIncidents.stream()
                .filter(i -> i.getSimilarityScore() > 0.2)
                .map(i -> AnalyzeResponse.SimilarIncident.builder()
                    .title(i.getTitle())
                    .resolution(i.getResolution())
                    .similarity(i.getSimilarityScore())
                    .build())
                .toList();

            return AnalyzeResponse.builder()
                .rootCause(node.path("root_cause").asText())
                .confidence(node.path("confidence").asInt(50))
                .severity(node.path("severity").asText("medium"))
                .errorCount(node.path("error_count").asInt())
                .warningCount(node.path("warning_count").asInt())
                .affectedServices(services)
                .mttrEstimateMinutes(node.has("mttr_estimate_minutes") && !node.get("mttr_estimate_minutes").isNull()
                    ? node.get("mttr_estimate_minutes").asInt() : null)
                .evidence(evidence)
                .suggestedActions(actions)
                .similarPastIncidents(similar)
                .build();

        } catch (Exception e) {
            throw new RuntimeException("Failed to parse LLM response as JSON: " + e.getMessage() + "\nRaw:\n" + raw, e);
        }
    }
}
