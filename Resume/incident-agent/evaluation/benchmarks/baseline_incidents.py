"""
Baseline incident benchmarks.
Run these after any model or prompt change to detect regressions.
"""

BASELINE_INCIDENTS = [
    {
        "id": "BENCH-001",
        "input": {
            "incident_id": "BENCH-001",
            "title": "Database connection pool exhausted",
            "description": "Payment service 500 errors. Connection pool at capacity.",
            "affected_service": "payment-service",
            "severity": "P1",
            "raw_logs": "ERROR: Connection pool exhausted 50/50\nERROR: Timeout acquiring connection",
        },
        "expected": {
            "root_cause_keywords": ["connection", "pool", "database"],
            "min_recommendations": 2,
            "expected_severity": "P1",
            "min_confidence": 0.6,
        }
    },
    {
        "id": "BENCH-002",
        "input": {
            "incident_id": "BENCH-002",
            "title": "API gateway 502 errors",
            "description": "Gateway returning 502. Upstream service not responding.",
            "affected_service": "api-gateway",
            "severity": "P2",
            "raw_logs": "ERROR: upstream connect error\nERROR: 502 Bad Gateway",
        },
        "expected": {
            "root_cause_keywords": ["upstream", "gateway", "timeout"],
            "min_recommendations": 2,
            "expected_severity": "P2",
            "min_confidence": 0.5,
        }
    },
]


def run_benchmarks() -> dict:
    """
    Run all baseline incidents and score against expected outputs.
    Returns pass/fail per benchmark with detailed scoring.
    """
    from main1crew import run_incident
    from rich.console import Console
    from rich.table import Table

    console = Console()
    results = []

    table = Table(title="Benchmark Results")
    table.add_column("ID",        style="cyan",  width=12)
    table.add_column("Status",    style="white", width=10)
    table.add_column("Confidence", style="white", width=12)
    table.add_column("Keywords",  style="white", width=10)
    table.add_column("Grade",     style="white", width=10)

    for bench in BASELINE_INCIDENTS:
        state = run_incident(**bench["input"], run_evaluation=True)
        expected = bench["expected"]

        report = (state.get("final_report") or "").lower()
        root_cause = (state.get("root_cause") or "").lower()
        full_text = report + root_cause

        # Check keyword presence
        keywords_found = sum(
            1 for kw in expected["root_cause_keywords"]
            if kw in full_text
        )
        keyword_score = keywords_found / len(expected["root_cause_keywords"])

        # Check recommendations
        recs = state.get("recommendations", [])
        recs_ok = len(recs) >= expected["min_recommendations"]

        # Check confidence
        confidence = state.get("confidence_score", 0)
        confidence_ok = confidence >= expected["min_confidence"]

        passed = keyword_score >= 0.6 and recs_ok and confidence_ok
        grade = state.get("evaluation", {}).get(
            "report_evaluation", {}
        ).get("grade", "N/A")

        status_color = "green" if passed else "red"
        table.add_row(
            bench["id"],
            f"[{status_color}]{'PASS' if passed else 'FAIL'}[/{status_color}]",
            f"{confidence:.2f}",
            f"{keywords_found}/{len(expected['root_cause_keywords'])}",
            grade,
        )

        results.append({
            "id": bench["id"],
            "passed": passed,
            "confidence": confidence,
            "keyword_score": keyword_score,
        })

    console.print(table)
    passed_count = sum(1 for r in results if r["passed"])
    console.print(
        f"\n[bold]Benchmark Score: {passed_count}/{len(results)}[/bold]\n"
    )
    return results


if __name__ == "__main__":
    run_benchmarks()