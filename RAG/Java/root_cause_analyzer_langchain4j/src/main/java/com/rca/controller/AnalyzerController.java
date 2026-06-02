package com.rca.controller;

import com.rca.model.AnalyzeRequest;
import com.rca.model.AnalyzeResponse;
import com.rca.rag.IncidentKnowledgeBase;
import com.rca.service.RootCauseOrchestrator;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/v1")
@RequiredArgsConstructor
@Tag(name = "Root Cause Analyzer", description = "LangChain4j-powered log analysis")
public class AnalyzerController {

    private final RootCauseOrchestrator orchestrator;
    private final IncidentKnowledgeBase knowledgeBase;

    @PostMapping("/analyze")
    @Operation(
        summary = "Analyze logs and identify root cause",
        description = "Ingests raw logs, runs LangChain4j RAG retrieval against past incidents, " +
                      "then calls Claude/GPT-4o to produce root cause + action items."
    )
    public ResponseEntity<AnalyzeResponse> analyze(@Valid @RequestBody AnalyzeRequest request) {
        return ResponseEntity.ok(orchestrator.analyze(request));
    }

    @PostMapping("/incidents")
    @Operation(
        summary = "Add a past incident to the LangChain4j EmbeddingStore",
        description = "Embeds the incident text and stores it in the vector store for future RAG retrieval."
    )
    public ResponseEntity<Map<String, String>> ingestIncident(
        @RequestParam String title,
        @RequestParam String description,
        @RequestParam String resolution
    ) {
        knowledgeBase.addIncident(title, description, resolution);
        return ResponseEntity.ok(Map.of("status", "ingested", "title", title));
    }

    @GetMapping("/health")
    @Operation(summary = "Health check")
    public ResponseEntity<Map<String, String>> health() {
        return ResponseEntity.ok(Map.of("status", "ok", "llm-framework", "LangChain4j"));
    }
}
