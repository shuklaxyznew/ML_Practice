"""
Data Ingestion Layer — filters, masks PII, summarizes logs before LLM processing.
"""

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class ProcessedLogs:
    raw_lines: List[str]
    error_lines: List[str]
    warning_lines: List[str]
    summary: str
    error_count: int
    warning_count: int
    service_name: str


# Basic PII masking patterns
PII_PATTERNS = [
    (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),                          # SSN
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),  # Email
    (r'\b(?:\d{4}[- ]?){3}\d{4}\b', '[CARD]'),                    # Credit card
    (r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[IP]'),                     # IP address
    (r'password[=:]\S+', 'password=[MASKED]'),                     # Passwords
    (r'token[=:]\S+', 'token=[MASKED]'),                           # Tokens
    (r'api[_-]?key[=:]\S+', 'api_key=[MASKED]'),                  # API keys
]

# Tags we care about
CRITICAL_TAGS = re.compile(r'\b(ERROR|CRITICAL|EXCEPTION|FATAL|SEVERE)\b', re.IGNORECASE)
WARNING_TAGS  = re.compile(r'\b(WARN|WARNING)\b', re.IGNORECASE)


class LogIngestionPipeline:

    def process(self, raw_logs: str, service_name: str = "unknown") -> ProcessedLogs:
        lines = [l.strip() for l in raw_logs.splitlines() if l.strip()]

        # Mask PII before anything else
        lines = [self._mask_pii(l) for l in lines]

        error_lines   = [l for l in lines if CRITICAL_TAGS.search(l)]
        warning_lines = [l for l in lines if WARNING_TAGS.search(l)]

        # Build a concise summary (max 30 lines) to stay within LLM context
        summary_lines = self._build_summary(error_lines, warning_lines, lines)

        return ProcessedLogs(
            raw_lines=lines,
            error_lines=error_lines,
            warning_lines=warning_lines,
            summary="\n".join(summary_lines),
            error_count=len(error_lines),
            warning_count=len(warning_lines),
            service_name=service_name,
        )

    def _mask_pii(self, line: str) -> str:
        for pattern, replacement in PII_PATTERNS:
            line = re.sub(pattern, replacement, line, flags=re.IGNORECASE)
        return line

    def _build_summary(self, errors, warnings, all_lines, max_lines=30) -> List[str]:
        """
        Priority: ERRORs first, then WARNs, then first/last info lines.
        Never exceed max_lines to control token cost.
        """
        seen = set()
        result = []

        for l in errors + warnings:
            if l not in seen:
                seen.add(l)
                result.append(l)
            if len(result) >= max_lines:
                break

        if len(result) < max_lines:
            for l in (all_lines[:5] + all_lines[-5:]):
                if l not in seen:
                    seen.add(l)
                    result.append(l)
                if len(result) >= max_lines:
                    break

        return result
