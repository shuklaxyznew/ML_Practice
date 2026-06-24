from evaluation.metrics import AgentEvaluator
from observability.logger import get_logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

logger = get_logger(__name__)
console = Console()


def evaluate_run(state: dict) -> dict:
    """
    Run full evaluation on a completed investigation.
    Call this after graph.invoke() returns.
    """
    evaluator = AgentEvaluator()

    report = state.get("final_report", "")
    tool_calls = state.get("tool_calls_made", [])

    report_eval = evaluator.evaluate_final_report(report, state)
    tool_eval = evaluator.evaluate_tool_effectiveness(tool_calls)

    # Print evaluation table
    table = Table(title="Agent Evaluation Results")
    table.add_column("Dimension",  style="cyan",  width=28)
    table.add_column("Score",      style="white", width=10)
    table.add_column("Weight",     style="white", width=10)

    dimension_labels = {
        "completeness":           "Report Completeness",
        "specificity":            "Answer Specificity",
        "actionability":          "Action Step Quality",
        "evidence_usage":         "Evidence / Tool Usage",
        "confidence_calibration": "Confidence Calibration",
    }

    weights = {
        "completeness": "30%",
        "specificity": "25%",
        "actionability": "20%",
        "evidence_usage": "15%",
        "confidence_calibration": "10%",
    }

    for key, label in dimension_labels.items():
        score = report_eval["scores"].get(key, 0)
        color = (
            "green" if score >= 0.7
            else "yellow" if score >= 0.5
            else "red"
        )
        table.add_row(
            label,
            f"[{color}]{score:.2f}[/{color}]",
            weights.get(key, ""),
        )

    console.print(table)

    grade_color = {
        "EXCELLENT": "green",
        "GOOD":      "cyan",
        "ACCEPTABLE": "yellow",
        "POOR":      "red",
    }.get(report_eval["grade"], "white")

    console.print(Panel(
        f"Overall Score: [bold]{report_eval['overall']:.2f}[/bold]\n"
        f"Grade: [{grade_color}]{report_eval['grade']}[/{grade_color}]\n"
        f"Tools used: {len(tool_eval.get('unique_tools', []))} unique "
        f"/ {tool_eval.get('total_calls', 0)} total calls",
        title="[bold]Evaluation Summary[/bold]",
        border_style=grade_color,
    ))

    return {
        "report_evaluation": report_eval,
        "tool_evaluation":   tool_eval,
    }