import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TestIncidentTool:
    """Tests for incident_lookup_tool and find_similar_incidents_tool."""

    def test_lookup_known_incident(self):
        from tools.incident_tool import incident_lookup_tool
        result = incident_lookup_tool.invoke({"incident_id": "INC-001"})
        assert "payment-service" in result
        assert "connection pool" in result.lower()

    def test_lookup_unknown_incident(self):
        from tools.incident_tool import incident_lookup_tool
        result = incident_lookup_tool.invoke({"incident_id": "INC-999"})
        assert "No incident found" in result
        assert "INC-001" in result  # shows available IDs

    def test_lookup_case_insensitive(self):
        from tools.incident_tool import incident_lookup_tool
        result = incident_lookup_tool.invoke({"incident_id": "inc-001"})
        assert "payment-service" in result  # lowercase should work

    def test_find_similar_by_description(self):
        from tools.incident_tool import find_similar_incidents_tool
        result = find_similar_incidents_tool.invoke({
            "description": "database connection timeout errors",
            "service": ""
        })
        assert "INC-001" in result  # most relevant match

    def test_find_similar_by_service(self):
        from tools.incident_tool import find_similar_incidents_tool
        result = find_similar_incidents_tool.invoke({
            "description": "service down",
            "service": "payment-service"
        })
        assert "INC-001" in result  # payment-service match

    def test_find_similar_no_match(self):
        from tools.incident_tool import find_similar_incidents_tool
        result = find_similar_incidents_tool.invoke({
            "description": "xyz quantum blockchain",
            "service": ""
        })
        assert "No similar incidents" in result


class TestLogParserTool:
    """Tests for log_parser_tool."""

    def test_parses_errors(self):
        from tools.log_parser_tool import log_parser_tool
        logs = (
            "2024-01-15 14:32:01 ERROR Database connection timeout\n"
            "2024-01-15 14:32:02 ERROR Failed to acquire connection\n"
            "2024-01-15 14:32:03 WARN Retry attempt 1/3"
        )
        result = log_parser_tool.invoke({"log_text": logs})
        assert "ERRORS" in result
        assert "WARNINGS" in result
        assert "2" in result  # 2 errors found

    def test_parses_timestamps(self):
        from tools.log_parser_tool import log_parser_tool
        logs = (
            "2024-01-15 14:32:01 ERROR First error\n"
            "2024-01-15 14:32:08 ERROR Last error"
        )
        result = log_parser_tool.invoke({"log_text": logs})
        assert "TIME RANGE" in result
        assert "14:32:01" in result
        assert "14:32:08" in result

    def test_empty_logs(self):
        from tools.log_parser_tool import log_parser_tool
        result = log_parser_tool.invoke({"log_text": ""})
        assert "No log text" in result

    def test_high_error_frequency_pattern(self):
        from tools.log_parser_tool import log_parser_tool
        # 6 errors should trigger high frequency pattern detection
        logs = "\n".join([
            f"2024-01-15 14:32:0{i} ERROR Error number {i}"
            for i in range(6)
        ])
        result = log_parser_tool.invoke({"log_text": logs})
        assert "High error frequency" in result


class TestKnowledgeRetriever:
    """Tests for ChromaDB retriever."""

    def test_retrieves_documents(self):
        from knowledge_base.retriever import retrieve
        results = retrieve("database connection pool", n_results=2)
        assert len(results) > 0
        assert "content" in results[0]
        assert "source" in results[0]
        assert "distance" in results[0]

    def test_retrieves_relevant_content(self):
        from knowledge_base.retriever import retrieve
        results = retrieve("database connection timeout errors", n_results=3)
        # The database runbook should be most relevant
        contents = [r["content"].lower() for r in results]
        assert any("database" in c or "connection" in c for c in contents)

    def test_collection_stats(self):
        from knowledge_base.retriever import get_collection_stats
        stats = get_collection_stats()
        assert stats["document_count"] >= 5
        assert "incident_knowledge" in stats["collection"]


class TestSessionMemory:
    """Tests for SessionMemory."""

    def test_add_and_retrieve_messages(self):
        from memory.session_memory import SessionMemory
        mem = SessionMemory(limit=10)
        mem.add_message("user", "What caused the outage?")
        mem.add_message("assistant", "Database connection pool exhausted.")
        messages = mem.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_memory_limit_enforced(self):
        from memory.session_memory import SessionMemory
        mem = SessionMemory(limit=3)
        for i in range(5):
            mem.add_message("user", f"Message {i}")
        messages = mem.get_messages()
        assert len(messages) == 3  # oldest dropped

    def test_tool_call_logging(self):
        from memory.session_memory import SessionMemory
        mem = SessionMemory()
        mem.add_tool_call("web_search", "payment errors", "Result text")
        calls = mem.get_tool_calls()
        assert len(calls) == 1
        assert calls[0]["tool"] == "web_search"

    def test_clear(self):
        from memory.session_memory import SessionMemory
        mem = SessionMemory()
        mem.add_message("user", "test")
        mem.clear()
        assert len(mem.get_messages()) == 0

    def test_context_summary(self):
        from memory.session_memory import SessionMemory
        mem = SessionMemory()
        mem.add_message("user", "test")
        mem.add_tool_call("web_search", "query", "result")
        summary = mem.get_context_summary()
        assert "1 messages" in summary
        assert "web_search" in summary


class TestHistoricalMemory:
    """Tests for HistoricalMemory SQLite persistence."""

    def setup_method(self):
        """Use a temp DB for each test."""
        import tempfile
        self.tmp = tempfile.mktemp(suffix=".db")

    def teardown_method(self):
        import gc
        import time
        gc.collect()
        time.sleep(0.1)
        if os.path.exists(self.tmp):
            try:
                os.remove(self.tmp)
            except PermissionError:
                pass  # acceptable on Windows

    def test_save_and_retrieve(self):
        from memory.historical_memory import HistoricalMemory
        mem = HistoricalMemory(db_path=self.tmp)
        mem.save_incident(
            incident_id="TEST-001",
            title="Test incident",
            service="test-service",
            severity="P2",
            root_cause="Test root cause",
            recommendations=["Fix A", "Fix B"],
            confidence=0.9,
            report="Full report text",
        )
        result = mem.get_incident("TEST-001")
        assert result is not None
        assert result["title"] == "Test incident"
        assert result["confidence"] == 0.9
        assert "Fix A" in result["recommendations"]

    def test_get_nonexistent(self):
        from memory.historical_memory import HistoricalMemory
        mem = HistoricalMemory(db_path=self.tmp)
        result = mem.get_incident("DOES-NOT-EXIST")
        assert result is None

    def test_count(self):
        from memory.historical_memory import HistoricalMemory
        mem = HistoricalMemory(db_path=self.tmp)
        assert mem.count() == 0
        mem.save_incident(
            "T-001", "t", "s", "P1", "rc",
            [], 0.8, "r"
        )
        assert mem.count() == 1

    def test_get_recent(self):
        from memory.historical_memory import HistoricalMemory
        mem = HistoricalMemory(db_path=self.tmp)
        for i in range(3):
            mem.save_incident(
                f"T-00{i}", f"title {i}", "svc",
                "P2", "rc", [], 0.8, "r"
            )
        recent = mem.get_recent(limit=2)
        assert len(recent) == 2