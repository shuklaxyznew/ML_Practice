package com.rca.service;

import com.rca.ingestion.LogIngestionPipeline;
import com.rca.model.AnalyzeRequest;
import com.rca.model.AnalyzeResponse;
import com.rca.model.Incident;
import com.rca.model.ProcessedLogs;
import com.rca.rag.IncidentKnowledgeBase;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * Orchestrates the full pipeline:
 *   1. Ingest & pre-process logs
 *   2. Retrieve similar past incidents (RAG)
 *   3. Run LLM reasoning chain
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class RootCauseOrchestrator {

    private final LogIngestionPipeline ingestionPipeline;
    private final IncidentKnowledgeBase knowledgeBase;
    private final AnalyzerService analyzerService;

    public AnalyzeResponse analyze(AnalyzeRequest request) {
        log.info("Starting analysis for service: {}", request.getServiceName());

        // Step 1: Ingest and pre-process
        ProcessedLogs processedLogs = ingestionPipeline.process(
            request.getLogs(),
            request.getServiceName()
        );
        log.info("Processed {} error lines, {} warning lines",
            processedLogs.getErrorCount(), processedLogs.getWarningCount());

        // Step 2: RAG — retrieve similar past incidents
        List<Incident> pastIncidents = knowledgeBase.search(processedLogs.getSummary(), 3);
        log.info("Retrieved {} similar past incidents", pastIncidents.size());

        // Step 3: LLM reasoning
        AnalyzeResponse response = analyzerService.analyze(processedLogs, pastIncidents, request.getMetrics());
        log.info("Analysis complete — root cause: {} (confidence: {}%)",
            response.getRootCause(), response.getConfidence());

        return response;
    }
}
