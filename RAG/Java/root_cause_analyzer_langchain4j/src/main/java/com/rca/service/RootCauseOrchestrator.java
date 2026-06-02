package com.rca.service;

import com.rca.ingestion.LogIngestionPipeline;
import com.rca.model.AnalyzeRequest;
import com.rca.model.AnalyzeResponse;
import com.rca.model.ProcessedLogs;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

/**
 * Orchestrates the full pipeline:
 *   Step 1 — Ingest & pre-process raw logs (PII masking, filtering)
 *   Step 2 — Call AnalyzerService which triggers LangChain4j RAG + LLM
 *   Step 3 — Return structured AnalyzeResponse
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class RootCauseOrchestrator {

    private final LogIngestionPipeline ingestionPipeline;
    private final AnalyzerService      analyzerService;

    public AnalyzeResponse analyze(AnalyzeRequest request) {
        log.info("Pipeline start — service: {}", request.getServiceName());

        // Step 1: Ingest
        ProcessedLogs processedLogs = ingestionPipeline.process(
            request.getLogs(),
            request.getServiceName()
        );
        log.info("Ingestion complete — {} errors, {} warnings",
            processedLogs.getErrorCount(), processedLogs.getWarningCount());

        // Step 2: LangChain4j RAG + LLM (handled inside AnalyzerService)
        AnalyzeResponse response = analyzerService.analyze(processedLogs, request.getMetrics());
        log.info("Analysis complete — root cause: '{}' (confidence: {}%)",
            response.getRootCause(), response.getConfidence());

        return response;
    }
}
