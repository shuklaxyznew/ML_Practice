from langchain.tools import tool
from pydantic import BaseModel, Field
from observability.logger import get_logger
import re

logger = get_logger(__name__)


class LogParserInput(BaseModel):
    log_text: str = Field(
        description="Raw log text to parse and extract error patterns from"
    )


@tool("parse_logs", args_schema=LogParserInput)
def log_parser_tool(log_text: str) -> str:
    """
    Parse raw log text and extract error patterns, stack traces,
    timestamps, and anomalies. Use this when raw logs are provided
    with the incident to identify what went wrong.
    """
    logger.info("Parsing logs")

    if not log_text or not log_text.strip():
        return "No log text provided."

    findings = {
        "errors": [],
        "warnings": [],
        "exceptions": [],
        "timestamps": [],
        "patterns": [],
    }

    lines = log_text.strip().split("\n")

    error_pattern = re.compile(r"(ERROR|FATAL|CRITICAL)", re.IGNORECASE)
    warning_pattern = re.compile(r"(WARN|WARNING)", re.IGNORECASE)
    exception_pattern = re.compile(r"(Exception|Traceback|Error:)", re.IGNORECASE)
    timestamp_pattern = re.compile(
        r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"
    )

    for line in lines:
        if error_pattern.search(line):
            findings["errors"].append(line.strip())
        if warning_pattern.search(line):
            findings["warnings"].append(line.strip())
        if exception_pattern.search(line):
            findings["exceptions"].append(line.strip())
        ts = timestamp_pattern.findall(line)
        findings["timestamps"].extend(ts)

    # Identify common patterns
    if len(findings["errors"]) > 5:
        findings["patterns"].append(
            f"High error frequency: {len(findings['errors'])} errors detected"
        )
    if findings["exceptions"]:
        findings["patterns"].append(
            f"Exceptions found: {len(findings['exceptions'])} exception traces"
        )

    output = [f"Log Analysis ({len(lines)} lines processed):\n"]

    if findings["errors"]:
        output.append(f"ERRORS ({len(findings['errors'])}):")
        output.extend(f"  {e}" for e in findings["errors"][:5])

    if findings["warnings"]:
        output.append(f"\nWARNINGS ({len(findings['warnings'])}):")
        output.extend(f"  {w}" for w in findings["warnings"][:3])

    if findings["exceptions"]:
        output.append(f"\nEXCEPTIONS ({len(findings['exceptions'])}):")
        output.extend(f"  {e}" for e in findings["exceptions"][:3])

    if findings["patterns"]:
        output.append(f"\nPATTERNS DETECTED:")
        output.extend(f"  {p}" for p in findings["patterns"])

    if findings["timestamps"]:
        output.append(
            f"\nTIME RANGE: {findings['timestamps'][0]} → {findings['timestamps'][-1]}"
        )

    return "\n".join(output)