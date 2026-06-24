from typing import TypedDict, Annotated, List, Optional
from operator import add
import time


class IncidentData(TypedDict):
    incident_id: str
    title: str
    description: str
    severity: Optional[str]
    affected_service: Optional[str]
    timestamp: Optional[str]
    raw_logs: Optional[str]


class AgentState(TypedDict):

    # ── Core incident ──
    incident: IncidentData

    # ── Workflow control ──
    current_node: str
    iteration_count: int
    max_iterations: int
    workflow_status: str          # "running" | "complete" | "failed"

    # ── Knowledge gathered ──
    search_results: Annotated[List[str], add]
    knowledge_context: Annotated[List[str], add]
    similar_incidents: Annotated[List[str], add]

    # ── Analysis ──
    root_cause: Optional[str]
    contributing_factors: Annotated[List[str], add]
    recommendations: Annotated[List[str], add]
    severity_assessment: Optional[str]

    # ── Reflection ──
    confidence_score: float
    reflection_notes: Annotated[List[str], add]
    needs_more_info: bool

    # ── Output ──
    final_report: Optional[str]
    action_items: Annotated[List[str], add]

    # ── Observability ──
    tool_calls_made: Annotated[List[str], add]
    errors_encountered: Annotated[List[str], add]
    start_time: float
    node_timings: Annotated[List[str], add]


def create_initial_state(
    incident_id: str,
    title: str,
    description: str,
    affected_service: str = "unknown",
    severity: str = "unknown",
    raw_logs: str = "",
) -> AgentState:
    """
    Factory function — always use this to create a fresh AgentState.
    Never construct AgentState manually elsewhere in the codebase.
    """
    return AgentState(
        incident=IncidentData(
            incident_id=incident_id,
            title=title,
            description=description,
            severity=severity,
            affected_service=affected_service,
            timestamp=str(time.time()),
            raw_logs=raw_logs,
        ),
        current_node="intake",
        iteration_count=0,
        max_iterations=10,
        workflow_status="running",
        search_results=[],
        knowledge_context=[],
        similar_incidents=[],
        root_cause=None,
        contributing_factors=[],
        recommendations=[],
        severity_assessment=None,
        confidence_score=0.0,
        reflection_notes=[],
        needs_more_info=False,
        final_report=None,
        action_items=[],
        tool_calls_made=[],
        errors_encountered=[],
        start_time=time.time(),
        node_timings=[],
    )