import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TestAgentState:
    """Tests for AgentState creation and structure."""

    def test_create_initial_state(self):
        from workflows.state import create_initial_state
        state = create_initial_state(
            incident_id="INT-001",
            title="Test incident",
            description="Test description",
            affected_service="test-service",
            severity="P2",
        )
        assert state["incident"]["incident_id"] == "INT-001"
        assert state["workflow_status"] == "running"
        assert state["confidence_score"] == 0.0
        assert state["iteration_count"] == 0
        assert state["needs_more_info"] == False
        assert isinstance(state["knowledge_context"], list)
        assert isinstance(state["recommendations"], list)

    def test_state_list_fields_start_empty(self):
        from workflows.state import create_initial_state
        state = create_initial_state("T-1", "t", "d")
        assert state["search_results"] == []
        assert state["tool_calls_made"] == []
        assert state["errors_encountered"] == []
        assert state["action_items"] == []

    def test_state_incident_data(self):
        from workflows.state import create_initial_state
        state = create_initial_state(
            incident_id="INT-002",
            title="DB down",
            description="Database is not responding",
            affected_service="db-service",
            severity="P1",
            raw_logs="ERROR: connection refused",
        )
        incident = state["incident"]
        assert incident["severity"] == "P1"
        assert incident["raw_logs"] == "ERROR: connection refused"
        assert incident["affected_service"] == "db-service"


class TestRouter:
    """Tests for conditional routing logic."""

    def test_routes_to_knowledge_when_needs_info(self):
        from workflows.router import route_after_coordinator
        from workflows.state import create_initial_state
        state = create_initial_state("T-1", "t", "d")
        state["needs_more_info"] = True
        result = route_after_coordinator(state)
        assert result == "knowledge_agent"

    def test_routes_direct_when_no_info_needed(self):
        from workflows.router import route_after_coordinator
        from workflows.state import create_initial_state
        state = create_initial_state("T-1", "t", "d")
        state["needs_more_info"] = False
        result = route_after_coordinator(state)
        assert result == "resolution_agent"

    def test_routes_to_output_when_confident(self):
        from workflows.router import route_after_reflection
        from workflows.state import create_initial_state
        state = create_initial_state("T-1", "t", "d")
        state["confidence_score"] = 0.9
        state["iteration_count"] = 1
        state["max_iterations"] = 10
        result = route_after_reflection(state)
        assert result == "output"

    def test_routes_to_retry_when_low_confidence(self):
        from workflows.router import route_after_reflection
        from workflows.state import create_initial_state
        state = create_initial_state("T-1", "t", "d")
        state["confidence_score"] = 0.2
        state["iteration_count"] = 1
        state["max_iterations"] = 10
        result = route_after_reflection(state)
        assert result == "coordinator"

    def test_forces_output_at_max_iterations(self):
        from workflows.router import route_after_reflection
        from workflows.state import create_initial_state
        state = create_initial_state("T-1", "t", "d")
        state["confidence_score"] = 0.1  # low confidence
        state["iteration_count"] = 10    # but at max
        state["max_iterations"] = 10
        result = route_after_reflection(state)
        assert result == "output"        # forces output anyway


class TestSettings:
    """Tests for configuration system."""

    def test_settings_load(self):
        from config.settings import settings
        assert settings.coordinator_model == "qwen2.5:3b"
        assert settings.resolution_model == "gemma3:4b"
        assert settings.confidence_threshold > 0
        assert settings.max_retries > 0

    def test_chroma_dir_is_string(self):
        from config.settings import settings
        assert isinstance(settings.chroma_persist_dir, str)
        assert "chroma_store" in settings.chroma_persist_dir

    def test_ollama_url(self):
        from config.settings import settings
        assert "localhost" in settings.ollama_base_url
        assert "11434" in settings.ollama_base_url