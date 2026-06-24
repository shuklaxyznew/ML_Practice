import time
import json
from pathlib import Path
from typing import Dict, List, Optional
from observability.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)

# Approximate token costs for local models
# On Ollama these are free — we track for production equivalents
TOKEN_COSTS = {
    "qwen2.5:3b":  {"input": 0.0, "output": 0.0},
    "gemma3:4b":   {"input": 0.0, "output": 0.0},
    # Production equivalents for reference
    "gpt-4o":      {"input": 0.0025, "output": 0.010},
    "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
}


class CostTracker:
    """
    Tracks token usage, latency, and tool calls per investigation.
    Writes metrics to a JSONL file for analysis.

    In production: ship these metrics to Prometheus / Datadog.
    """

    def __init__(self, incident_id: str):
        self.incident_id = incident_id
        self.start_time = time.time()
        self.llm_calls: List[Dict] = []
        self.tool_calls: List[Dict] = []
        self.node_timings: List[Dict] = []
        self._current_node_start: Optional[float] = None
        self._current_node: Optional[str] = None

    def record_llm_call(
        self,
        model: str,
        node: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0.0,
    ) -> None:
        costs = TOKEN_COSTS.get(model, {"input": 0.0, "output": 0.0})
        estimated_cost = (
            prompt_tokens * costs["input"] / 1000 +
            completion_tokens * costs["output"] / 1000
        )
        self.llm_calls.append({
            "model":             model,
            "node":              node,
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":      prompt_tokens + completion_tokens,
            "latency_ms":        latency_ms,
            "estimated_cost_usd": estimated_cost,
            "timestamp":         time.time(),
        })
        logger.debug(
            f"LLM call: {model} | {prompt_tokens}+{completion_tokens} tokens "
            f"| {latency_ms:.0f}ms"
        )

    def record_tool_call(
        self,
        tool_name: str,
        success: bool,
        latency_ms: float,
        node: str = "",
    ) -> None:
        self.tool_calls.append({
            "tool":       tool_name,
            "success":    success,
            "latency_ms": latency_ms,
            "node":       node,
            "timestamp":  time.time(),
        })

    def start_node(self, node_name: str) -> None:
        self._current_node = node_name
        self._current_node_start = time.time()

    def end_node(self, node_name: str) -> None:
        if self._current_node_start:
            duration = (time.time() - self._current_node_start) * 1000
            self.node_timings.append({
                "node":        node_name,
                "duration_ms": round(duration, 2),
            })
            self._current_node_start = None

    def get_summary(self) -> Dict:
        total_tokens = sum(c["total_tokens"] for c in self.llm_calls)
        total_cost = sum(c["estimated_cost_usd"] for c in self.llm_calls)
        total_latency = (time.time() - self.start_time) * 1000
        tool_success_rate = (
            sum(1 for t in self.tool_calls if t["success"]) /
            len(self.tool_calls) if self.tool_calls else 0
        )

        return {
            "incident_id":       self.incident_id,
            "total_duration_ms": round(total_latency, 2),
            "llm_calls":         len(self.llm_calls),
            "total_tokens":      total_tokens,
            "estimated_cost_usd": round(total_cost, 6),
            "tool_calls":        len(self.tool_calls),
            "tool_success_rate": round(tool_success_rate, 3),
            "node_timings":      self.node_timings,
            "slowest_node": (
                max(self.node_timings, key=lambda x: x["duration_ms"])
                if self.node_timings else None
            ),
        }

    def save(self) -> None:
        summary = self.get_summary()
        log_path = Path(settings.log_file).parent / "metrics.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")

        logger.info(
            f"Metrics saved | tokens: {summary['total_tokens']} | "
            f"tools: {summary['tool_calls']} | "
            f"duration: {summary['total_duration_ms']:.0f}ms"
        )