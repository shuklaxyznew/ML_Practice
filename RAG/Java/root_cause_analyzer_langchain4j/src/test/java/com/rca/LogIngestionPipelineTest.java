package com.rca;

import com.rca.ingestion.LogIngestionPipeline;
import com.rca.model.ProcessedLogs;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class LogIngestionPipelineTest {

    private LogIngestionPipeline pipeline;

    @BeforeEach
    void setUp() { pipeline = new LogIngestionPipeline(); }

    @Test
    void countsErrorsAndWarnings() {
        String logs = "ERROR DB connection failed\nWARN low memory\nINFO ok";
        ProcessedLogs r = pipeline.process(logs, "svc");
        assertThat(r.getErrorCount()).isEqualTo(1);
        assertThat(r.getWarningCount()).isEqualTo(1);
    }

    @Test
    void masksEmail() {
        ProcessedLogs r = pipeline.process("ERROR user john@example.com failed", "svc");
        assertThat(r.getSummary()).doesNotContain("john@example.com");
        assertThat(r.getSummary()).contains("[EMAIL]");
    }

    @Test
    void masksPassword() {
        ProcessedLogs r = pipeline.process("WARN password=supersecret123", "svc");
        assertThat(r.getSummary()).doesNotContain("supersecret123");
        assertThat(r.getSummary()).contains("[MASKED]");
    }

    @Test
    void masksApiKey() {
        ProcessedLogs r = pipeline.process("ERROR api_key=sk-abc123", "svc");
        assertThat(r.getSummary()).doesNotContain("sk-abc123");
    }

    @Test
    void emptyLogs() {
        ProcessedLogs r = pipeline.process("", "svc");
        assertThat(r.getErrorCount()).isZero();
        assertThat(r.getWarningCount()).isZero();
    }

    @Test
    void summaryIsCappedAt30Lines() {
        String logs = "ERROR line\n".repeat(100);
        ProcessedLogs r = pipeline.process(logs, "svc");
        assertThat(r.getSummary().split("\n").length).isLessThanOrEqualTo(30);
    }

    @Test
    void criticalAndFatalCountAsErrors() {
        ProcessedLogs r = pipeline.process("CRITICAL down\nFATAL crash", "svc");
        assertThat(r.getErrorCount()).isEqualTo(2);
    }

    @Test
    void caseInsensitiveDetection() {
        ProcessedLogs r = pipeline.process("error bad\nwarning low", "svc");
        assertThat(r.getErrorCount()).isEqualTo(1);
        assertThat(r.getWarningCount()).isEqualTo(1);
    }
}
