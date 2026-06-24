import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TestFullWorkflow:
    """
    End-to-end tests that run the full LangGraph pipeline.
    These are slow (2-5 min each) — run manually, not in CI.
    Requires Ollama running with both models pulled.
    """

    def test_complete_incident_investigation(self):
        from main import run_incident
        state = run_incident(
            incident_id="E2E-001",
            title="API service returning 503 errors",
            description=(
                "The API service started returning 503 errors. "
                "CPU utilization spiked to 95%. Response times degraded."
            ),
            affected_service="api-service",
            severity="P2",
        )
        # Workflow completed
        assert state["workflow_status"] == "complete"
        # Report was generated
        assert state["final_report"] is not None
        assert len(state["final_report"]) > 100
        # Agents ran
        assert state["iteration_count"] >= 1
        # Confidence was scored
        assert state["confidence_score"] > 0

    def test_state_populated_after_run(self):
        from main import run_incident
        state = run_incident(
            incident_id="E2E-002",
            title="Memory leak in worker service",
            description="Worker service memory growing unbounded. OOM errors in logs.",
            affected_service="worker-service",
            severity="P2",
            raw_logs=(
                "2024-01-15 10:00:00 ERROR OutOfMemoryError: Java heap space\n"
                "2024-01-15 10:00:05 WARN Memory usage at 95%\n"
                "2024-01-15 10:00:10 ERROR Failed to allocate memory"
            ),
        )
        assert state["workflow_status"] == "complete"
        assert len(state["knowledge_context"]) > 0
        assert len(state["tool_calls_made"]) > 0