"""
Unit tests for AI Root Cause Analyzer
Run with: pytest tests/ -v
"""

import pytest
from app.ingestion import LogIngestionPipeline
from app.rag import IncidentKnowledgeBase, BUILTIN_INCIDENTS


# ── Ingestion Tests ──────────────────────────────────────────────────────────

class TestLogIngestionPipeline:
    def setup_method(self):
        self.pipeline = LogIngestionPipeline()

    def test_basic_processing(self):
        logs = "ERROR 2026-05-06 — DB connection failed\nWARN low memory\nINFO startup complete"
        result = self.pipeline.process(logs, "test-svc")
        assert result.error_count == 1
        assert result.warning_count == 1
        assert result.service_name == "test-svc"

    def test_pii_masking_email(self):
        logs = "ERROR user john.doe@example.com failed login"
        result = self.pipeline.process(logs)
        assert "john.doe@example.com" not in result.summary
        assert "[EMAIL]" in result.summary

    def test_pii_masking_password(self):
        logs = "WARN connection string: password=supersecret123"
        result = self.pipeline.process(logs)
        assert "supersecret123" not in result.summary
        assert "[MASKED]" in result.summary

    def test_pii_masking_api_key(self):
        logs = "ERROR api_key=sk-1234567890abcdef not valid"
        result = self.pipeline.process(logs)
        assert "sk-1234567890abcdef" not in result.summary

    def test_empty_logs(self):
        result = self.pipeline.process("", "empty-svc")
        assert result.error_count == 0
        assert result.warning_count == 0

    def test_summary_max_lines(self):
        # 100 lines of errors — summary should be capped
        logs = "\n".join([f"ERROR line {i} — something failed" for i in range(100)])
        result = self.pipeline.process(logs)
        summary_lines = result.summary.split("\n")
        assert len(summary_lines) <= 30

    def test_critical_tag_counts_as_error(self):
        logs = "CRITICAL service is down\nFATAL unrecoverable error"
        result = self.pipeline.process(logs)
        assert result.error_count == 2

    def test_case_insensitive_tag_detection(self):
        logs = "error something bad\nwarning low disk"
        result = self.pipeline.process(logs)
        assert result.error_count == 1
        assert result.warning_count == 1


# ── RAG Knowledge Base Tests ──────────────────────────────────────────────────

class TestIncidentKnowledgeBase:
    def setup_method(self):
        self.kb = IncidentKnowledgeBase()

    def test_fallback_search_db(self):
        results = self.kb._fallback_search("database connection pool hikari jdbc")
        assert len(results) > 0
        assert any("pool" in r.title.lower() or "db" in r.title.lower() for r in results)

    def test_fallback_search_oom(self):
        results = self.kb._fallback_search("OutOfMemoryError heap java killed")
        assert len(results) > 0

    def test_fallback_search_no_match(self):
        results = self.kb._fallback_search("completely unrelated query xyz123")
        assert len(results) == 0

    def test_fallback_returns_max_3(self):
        results = self.kb._fallback_search("error memory connection timeout crash")
        assert len(results) <= 3

    def test_similarity_score_between_0_and_1(self):
        results = self.kb._fallback_search("timeout circuit breaker payment")
        for r in results:
            assert 0.0 <= r.similarity_score <= 1.0

    def test_builtin_incidents_have_required_fields(self):
        for inc in BUILTIN_INCIDENTS:
            assert "title" in inc
            assert "description" in inc
            assert "resolution" in inc


# ── Integration smoke test ────────────────────────────────────────────────────

class TestIntegration:
    def test_ingestion_to_rag_pipeline(self):
        pipeline = LogIngestionPipeline()
        kb = IncidentKnowledgeBase()

        logs = """
        ERROR 2026-05-06 12:01:23 [api] HikariPool-1 Connection is not available, timeout 30000ms
        CRITICAL 2026-05-06 12:01:25 [api] Health check FAILED: DataSource failure
        WARN 2026-05-06 12:00:58 [api] Pool usage 95%: 19/20 connections active
        """
        processed = pipeline.process(logs, "api-service")
        assert processed.error_count >= 1

        incidents = kb.search(processed.summary)
        # Should match DB pool incident
        assert isinstance(incidents, list)
