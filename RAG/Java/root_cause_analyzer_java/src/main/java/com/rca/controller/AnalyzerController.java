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
@Tag(name = "Root Cause Analyzer", description = "AI-powered log analysis and root cause identification")
public class AnalyzerController {

    private final RootCauseOrchestrator orchestrator;
    private final IncidentKnowledgeBase knowledgeBase;

    @PostMapping("/analyze")
    @Operation(summary = "Analyze logs and identify root cause",
               description = "Ingests raw logs/metrics, retrieves similar past incidents via RAG, and uses LLM to identify root cause + suggest fixes.")
    public ResponseEntity<AnalyzeResponse> analyze(@Valid @RequestBody AnalyzeRequest request) {
        AnalyzeResponse response = orchestrator.analyze(request);
        return ResponseEntity.ok(response);
    }

    @PostMapping("/incidents")
    @Operation(summary = "Add a past incident to the knowledge base",
               description = "Embeds a post-mortem or JIRA ticket into the RAG vector store for future retrieval.")
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
        return ResponseEntity.ok(Map.of("status", "ok", "service", "root-cause-analyzer"));
    }
}
