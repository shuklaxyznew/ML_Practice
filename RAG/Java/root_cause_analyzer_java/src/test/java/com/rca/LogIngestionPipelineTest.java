package com.rca;

import com.rca.ingestion.LogIngestionPipeline;
import com.rca.model.ProcessedLogs;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class LogIngestionPipelineTest {

    private LogIngestionPipeline pipeline;

    @BeforeEach
    void setUp() {
        pipeline = new LogIngestionPipeline();
    }

    @Test
    void basicProcessing_countsErrorsAndWarnings() {
        String logs = """
            ERROR 2026-05-06 — DB connection failed
            WARN low memory
            INFO startup complete
            """;
        ProcessedLogs result = pipeline.process(logs, "test-svc");

        assertThat(result.getErrorCount()).isEqualTo(1);
        assertThat(result.getWarningCount()).isEqualTo(1);
        assertThat(result.getServiceName()).isEqualTo("test-svc");
    }

    @Test
    void piiMasking_removesEmail() {
        String logs = "ERROR user john.doe@example.com failed login";
        ProcessedLogs result = pipeline.process(logs, "svc");

        assertThat(result.getSummary()).doesNotContain("john.doe@example.com");
        assertThat(result.getSummary()).contains("[EMAIL]");
    }

    @Test
    void piiMasking_removesPassword() {
        String logs = "WARN connection string: password=supersecret123";
        ProcessedLogs result = pipeline.process(logs, "svc");

        assertThat(result.getSummary()).doesNotContain("supersecret123");
        assertThat(result.getSummary()).contains("[MASKED]");
    }

    @Test
    void piiMasking_removesApiKey() {
        String logs = "ERROR api_key=sk-1234567890abcdef not valid";
        ProcessedLogs result = pipeline.process(logs, "svc");

        assertThat(result.getSummary()).doesNotContain("sk-1234567890abcdef");
    }

    @Test
    void emptyLogs_returnsZeroCounts() {
        ProcessedLogs result = pipeline.process("", "empty-svc");

        assertThat(result.getErrorCount()).isZero();
        assertThat(result.getWarningCount()).isZero();
    }

    @Test
    void largeLogs_summaryIsCappedAt30Lines() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 100; i++) {
            sb.append("ERROR line ").append(i).append(" — something failed\n");
        }
        ProcessedLogs result = pipeline.process(sb.toString(), "svc");

        long lineCount = result.getSummary().lines().count();
        assertThat(lineCount).isLessThanOrEqualTo(30);
    }

    @Test
    void criticalAndFatalTags_countedAsErrors() {
        String logs = "CRITICAL service is down\nFATAL unrecoverable error";
        ProcessedLogs result = pipeline.process(logs, "svc");

        assertThat(result.getErrorCount()).isEqualTo(2);
    }

    @Test
    void caseInsensitiveTagDetection() {
        String logs = "error something bad\nwarning low disk";
        ProcessedLogs result = pipeline.process(logs, "svc");

        assertThat(result.getErrorCount()).isEqualTo(1);
        assertThat(result.getWarningCount()).isEqualTo(1);
    }
}
