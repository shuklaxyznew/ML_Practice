"""
Analyzer Engine — uses LangChain + LLM to identify root cause and suggest fixes.
Supports Anthropic Claude (default) and OpenAI GPT-4o.
"""

import os
import json
from typing import List, Optional
from dataclasses import dataclass, asdict

from app.ingestion import ProcessedLogs
from app.rag import Incident
from prompts.system_prompt import SYSTEM_PROMPT


@dataclass
class AnalysisResult:
    root_cause: str
    confidence: int
    severity: str
    evidence: List[dict]
    suggested_actions: List[dict]
    similar_past_incidents: List[dict]
    mttr_estimate_minutes: Optional[int] = None


class RootCauseAnalyzer:
    """
    Orchestrates the full analysis pipeline:
      1. Format processed logs + RAG context into a prompt
      2. Call the LLM
      3. Parse and return structured results
    """

    def __init__(self):
        self.llm = self._init_llm()

    def _init_llm(self):
        """
        Initialize LLM client. Prefers Anthropic Claude, falls back to OpenAI.
        Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your environment.
        """
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if anthropic_key:
            try:
                from langchain_anthropic import ChatAnthropic
                return ChatAnthropic(
                    model="claude-sonnet-4-20250514",
                    api_key=anthropic_key,
                    temperature=0,
                    max_tokens=1500
                )
            except ImportError:
                pass

        if openai_key:
            try:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model="gpt-4o",
                    api_key=openai_key,
                    temperature=0,
                    max_tokens=1500
                )
            except ImportError:
                pass

        # Fallback: raise a clear error
        raise EnvironmentError(
            "No LLM configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your .env file."
        )

    def analyze(
        self,
        processed_logs: ProcessedLogs,
        past_incidents: List[Incident],
        metrics: Optional[str] = None
    ) -> AnalysisResult:

        # Build RAG context string
        rag_context = self._format_rag_context(past_incidents)

        # Build the user prompt
        user_prompt = self._build_user_prompt(processed_logs, rag_context, metrics)

        # Call LLM
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        response = self.llm.invoke(messages)
        raw = response.content

        # Parse JSON response
        parsed = self._parse_response(raw)

        # Attach similar incidents to result
        parsed["similar_past_incidents"] = [
            {
                "title": inc.title,
                "resolution": inc.resolution,
                "similarity": inc.similarity_score
            }
            for inc in past_incidents
            if inc.similarity_score > 0.2
        ]

        return AnalysisResult(**parsed)

    def _format_rag_context(self, incidents: List[Incident]) -> str:
        if not incidents:
            return "No similar past incidents found."
        lines = []
        for i, inc in enumerate(incidents, 1):
            lines.append(f"{i}. [{inc.similarity_score:.0%} match] {inc.title}")
            lines.append(f"   Resolution: {inc.resolution}")
        return "\n".join(lines)

    def _build_user_prompt(
        self,
        logs: ProcessedLogs,
        rag_context: str,
        metrics: Optional[str]
    ) -> str:
        parts = [
            f"Service: {logs.service_name}",
            f"Errors: {logs.error_count} | Warnings: {logs.warning_count}",
            "",
            "=== CURRENT LOGS ===",
            logs.summary,
        ]
        if metrics:
            parts += ["", "=== METRICS ===", metrics]
        parts += ["", "=== SIMILAR PAST INCIDENTS ===", rag_context]
        return "\n".join(parts)

    def _parse_response(self, raw: str) -> dict:
        """Strip markdown fences and parse JSON."""
        clean = raw.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:])
        if clean.endswith("```"):
            clean = "\n".join(clean.split("\n")[:-1])
        clean = clean.strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned non-JSON response: {e}\n\nRaw:\n{raw}")
