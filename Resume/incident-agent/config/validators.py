import re
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class IncidentValidator(BaseModel):
    """
    Validates and sanitizes incident input before agents see it.
    Prevents prompt injection and ensures data quality.
    """
    incident_id: str
    title: str = Field(min_length=5, max_length=200)
    description: str = Field(min_length=10, max_length=5000)
    affected_service: str = Field(min_length=2, max_length=100)
    severity: str = Field(default="unknown")
    raw_logs: Optional[str] = Field(default="", max_length=10000)

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        allowed = {"P1", "P2", "P3", "P4", "unknown"}
        if v.upper() in allowed:
            return v.upper()
        return "unknown"

    @field_validator("title", "description")
    @classmethod
    def sanitize_prompt_injection(cls, v: str) -> str:
        """
        Remove common prompt injection patterns.
        Prevents malicious input from hijacking agent behavior.
        """
        injection_patterns = [
            r"ignore previous instructions",
            r"ignore all instructions",
            r"you are now",
            r"act as",
            r"system prompt",
            r"<\|im_start\|>",
            r"<\|im_end\|>",
        ]
        v_lower = v.lower()
        for pattern in injection_patterns:
            if re.search(pattern, v_lower):
                raise ValueError(
                    f"Invalid input: potential prompt injection detected"
                )
        return v.strip()

    @field_validator("raw_logs")
    @classmethod
    def sanitize_logs(cls, v: str) -> str:
        if not v:
            return ""
        # Truncate logs to prevent context window overflow
        max_lines = 100
        lines = v.split("\n")
        if len(lines) > max_lines:
            return "\n".join(lines[:max_lines]) + f"\n[{len(lines) - max_lines} lines truncated]"
        return v