import re
import time
from typing import Dict, List
from observability.logger import get_logger

logger = get_logger(__name__)


class AgentEvaluator:
    """
    Evaluates the quality of agent outputs without human review.
    Uses heuristic scoring — production systems add LLM-as-judge.
    """

    def evaluate_final_report(self, report: str, state: Dict) -> Dict:
        """Score a completed incident report on multiple dimensions."""

        scores = {}

        # 1. Completeness — does it have all required sections?
        required_sections = [
            "ROOT CAUSE",
            "CONTRIBUTING FACTORS",
            "IMMEDIATE ACTIONS",
            "PREVENTIVE MEASURES",
            "SEVERITY",
            "CONFIDENCE",
        ]
        found = sum(1 for s in required_sections if s in report.upper())
        scores["completeness"] = round(found / len(required_sections), 2)

        # 2. Specificity — is the root cause specific or vague?
        vague_phrases = [
            "unknown", "unclear", "possibly", "might be",
            "could be", "not sure", "investigation needed"
        ]
        vague_count = sum(
            1 for p in vague_phrases if p in report.lower()
        )
        scores["specificity"] = round(max(0, 1.0 - vague_count * 0.2), 2)

        # 3. Actionability — does it have numbered action steps?
        action_steps = re.findall(r"^\d+\.", report, re.MULTILINE)
        scores["actionability"] = min(1.0, len(action_steps) * 0.2)

        # 4. Evidence usage — did the agent use tools?
        tool_calls = state.get("tool_calls_made", [])
        scores["evidence_usage"] = min(1.0, len(tool_calls) * 0.15)

        # 5. Confidence calibration — is confidence reasonable?
        confidence = state.get("confidence_score", 0)
        # Penalize both very low (< 0.3) and suspiciously high (1.0)
        if confidence < 0.3:
            scores["confidence_calibration"] = 0.3
        elif confidence == 1.0:
            scores["confidence_calibration"] = 0.8
        else:
            scores["confidence_calibration"] = confidence

        # Overall weighted score
        weights = {
            "completeness":          0.30,
            "specificity":           0.25,
            "actionability":         0.20,
            "evidence_usage":        0.15,
            "confidence_calibration": 0.10,
        }
        overall = sum(scores[k] * weights[k] for k in weights)
        scores["overall"] = round(overall, 3)

        # Grade
        if overall >= 0.85:
            grade = "EXCELLENT"
        elif overall >= 0.70:
            grade = "GOOD"
        elif overall >= 0.50:
            grade = "ACCEPTABLE"
        else:
            grade = "POOR"

        return {
            "scores":    scores,
            "grade":     grade,
            "overall":   overall,
            "timestamp": time.time(),
        }

    def evaluate_tool_effectiveness(
        self, tool_calls: List[str]
    ) -> Dict:
        """Analyze tool usage patterns."""
        if not tool_calls:
            return {"effectiveness": 0.0, "notes": "No tools used"}

        unique_tools = set(tool_calls)
        total_calls = len(tool_calls)
        unique_count = len(unique_tools)

        # Penalize repetitive tool calls (same tool > 3 times)
        from collections import Counter
        call_counts = Counter(tool_calls)
        repetitive = sum(
            1 for count in call_counts.values() if count > 3
        )

        effectiveness = min(1.0, unique_count * 0.25) - repetitive * 0.1

        return {
            "total_calls":   total_calls,
            "unique_tools":  list(unique_tools),
            "call_counts":   dict(call_counts),
            "repetitive":    repetitive,
            "effectiveness": round(max(0, effectiveness), 3),
        }