import os
import warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"
warnings.filterwarnings("ignore", message="Failed to send telemetry")

from workflows.graph import build_graph
from workflows.state import create_initial_state
from observability.logger import get_logger
from observability.cost_tracker import CostTracker
from evaluation.evaluator import evaluate_run
from rich.console import Console
from rich.panel import Panel

logger = get_logger(__name__)
console = Console()


def run_incident(
    incident_id: str,
    title: str,
    description: str,
    affected_service: str = "unknown",
    severity: str = "unknown",
    raw_logs: str = "",
    run_evaluation: bool = True,
) -> dict:

    console.print(Panel(
        f"[bold]Incident:[/bold] {title}\n"
        f"[bold]Service:[/bold]  {affected_service}\n"
        f"[bold]Severity:[/bold] {severity}",
        title=f"[bold blue]Enterprise Incident Agent — {incident_id}[/bold blue]",
        border_style="blue",
    ))

    logger.info(f"Starting investigation: {incident_id}")

    # Start cost tracking
    tracker = CostTracker(incident_id)
    tracker.start_node("full_workflow")

    graph = build_graph()

    initial_state = create_initial_state(
        incident_id=incident_id,
        title=title,
        description=description,
        affected_service=affected_service,
        severity=severity,
        raw_logs=raw_logs,
    )

    final_state = graph.invoke(initial_state)

    tracker.end_node("full_workflow")
    tracker.save()

    console.print(Panel(
        final_state.get("final_report", "No report generated"),
        title="[bold green]Investigation Complete[/bold green]",
        border_style="green",
    ))

    # Run evaluation
    if run_evaluation:
        console.print("\n")
        eval_results = evaluate_run(final_state)
        final_state["evaluation"] = eval_results

    return final_state


if __name__ == "__main__":
    run_incident(
        incident_id="INC-TEST-001",
        title="Payment service returning 500 errors",
        description=(
            "The payment service started returning HTTP 500 errors at 14:32 UTC. "
            "Error rate jumped from 0.1% to 45% within 5 minutes. "
            "Database connection timeouts observed in logs."
        ),
        affected_service="payment-service",
        severity="P1",
        raw_logs=(
            "2024-01-15 14:32:01 ERROR Database connection timeout after 30s\n"
            "2024-01-15 14:32:02 ERROR Failed to acquire connection from pool\n"
            "2024-01-15 14:32:03 ERROR Connection pool exhausted: 50/50 connections in use\n"
            "2024-01-15 14:32:05 WARN Retry attempt 1/3 for transaction TX-8821\n"
            "2024-01-15 14:32:08 ERROR Transaction TX-8821 failed after 3 retries\n"
        ),
    )