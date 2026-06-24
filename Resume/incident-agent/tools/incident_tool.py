from langchain.tools import tool
from pydantic import BaseModel, Field
from observability.logger import get_logger
import json
import os

logger = get_logger(__name__)

SAMPLE_INCIDENTS = {
    "INC-001": {
        "id": "INC-001",
        "title": "Database connection pool exhausted",
        "service": "payment-service",
        "severity": "P1",
        "root_cause": "Connection leak in payment processor due to unclosed transactions",
        "resolution": "Restarted service, increased pool size, added connection timeout",
        "duration_minutes": 45,
    },
    "INC-002": {
        "id": "INC-002",
        "title": "API gateway returning 502 errors",
        "service": "api-gateway",
        "severity": "P2",
        "root_cause": "Upstream service timeout causing gateway to return 502",
        "resolution": "Scaled upstream service, adjusted timeout thresholds",
        "duration_minutes": 20,
    },
    "INC-003": {
        "id": "INC-003",
        "title": "Memory leak in recommendation engine",
        "service": "recommendation-service",
        "severity": "P2",
        "root_cause": "Unbounded cache growth without eviction policy",
        "resolution": "Deployed cache eviction fix, restarted pods",
        "duration_minutes": 60,
    },
}


class IncidentLookupInput(BaseModel):
    incident_id: str = Field(
        description="The incident ID to look up, e.g. INC-001"
    )


class SimilarIncidentInput(BaseModel):
    description: str = Field(
        description="Description of the current incident to find similar past incidents"
    )
    service: str = Field(
        default="",
        description="Affected service name to narrow the search"
    )


@tool("incident_lookup", args_schema=IncidentLookupInput)
def incident_lookup_tool(incident_id: str) -> str:
    """
    Look up a specific past incident by ID. Returns full details
    including root cause and resolution. Use when you have a
    specific incident ID to reference.
    """
    logger.info(f"Incident lookup: {incident_id}")
    incident = SAMPLE_INCIDENTS.get(incident_id.upper())

    if not incident:
        return f"No incident found with ID {incident_id}. Available: {list(SAMPLE_INCIDENTS.keys())}"

    return json.dumps(incident, indent=2)


@tool("find_similar_incidents", args_schema=SimilarIncidentInput)
def find_similar_incidents_tool(description: str, service: str = "") -> str:
    """
    Find past incidents similar to the current one based on description
    and affected service. Use this to identify patterns and proven
    resolutions from historical data.
    """
    logger.info(f"Finding similar incidents for: {description[:50]}")

    keywords = description.lower().split()
    matches = []

    for inc_id, incident in SAMPLE_INCIDENTS.items():
        score = 0
        text = f"{incident['title']} {incident['root_cause']}".lower()

        for kw in keywords:
            if kw in text:
                score += 1

        if service and service.lower() in incident["service"]:
            score += 2

        if score > 0:
            matches.append((score, incident))

    matches.sort(key=lambda x: x[0], reverse=True)

    if not matches:
        return "No similar incidents found in history."

    formatted = []
    for score, inc in matches[:3]:
        formatted.append(
            f"Incident: {inc['id']} (relevance: {score})\n"
            f"Title: {inc['title']}\n"
            f"Root Cause: {inc['root_cause']}\n"
            f"Resolution: {inc['resolution']}\n"
            f"Duration: {inc['duration_minutes']} minutes"
        )

    return "\n\n".join(formatted)