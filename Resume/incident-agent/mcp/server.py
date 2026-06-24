from fastmcp import FastMCP
from observability.logger import get_logger
from config.settings import settings
import json
import time

logger = get_logger(__name__)

# Create the MCP server instance
mcp = FastMCP(
    name="Incident Resolution Tools",
    version="1.0.0",
)


@mcp.tool()
def search_knowledge_base(query: str, n_results: int = 3) -> str:
    """
    Search the internal knowledge base for runbooks and documentation.
    Returns relevant documents ranked by semantic similarity.

    Args:
        query: Natural language search query
        n_results: Number of results to return (default 3)
    """
    try:
        from knowledge_base.retriever import retrieve
        results = retrieve(query, n_results=n_results)
        if not results:
            return json.dumps({"status": "empty", "results": []})
        return json.dumps({
            "status": "ok",
            "count": len(results),
            "results": results,
        })
    except Exception as e:
        logger.error(f"MCP knowledge search failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def get_incident_details(incident_id: str) -> str:
    """
    Retrieve full details of a past incident by ID.

    Args:
        incident_id: Incident identifier e.g. INC-001
    """
    try:
        from tools.incident_tool import SAMPLE_INCIDENTS
        incident = SAMPLE_INCIDENTS.get(incident_id.upper())
        if not incident:
            return json.dumps({
                "status": "not_found",
                "available": list(SAMPLE_INCIDENTS.keys()),
            })
        return json.dumps({"status": "ok", "incident": incident})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def analyze_logs(log_text: str) -> str:
    """
    Parse and analyze raw log text to extract error patterns.

    Args:
        log_text: Raw log content to analyze
    """
    try:
        from tools.log_parser_tool import log_parser_tool
        result = log_parser_tool.invoke({"log_text": log_text})
        return json.dumps({"status": "ok", "analysis": result})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def check_service_health(service_name: str) -> str:
    """
    Check the health status of a named service.
    In production this would ping real endpoints.

    Args:
        service_name: Name of the service to check
    """
    # Simulated health check — Phase 8 replaces with real HTTP pings
    health_data = {
        "payment-service":        {"status": "degraded",  "latency_ms": 2400, "error_rate": 0.45},
        "api-gateway":            {"status": "healthy",   "latency_ms": 45,   "error_rate": 0.001},
        "recommendation-service": {"status": "healthy",   "latency_ms": 120,  "error_rate": 0.002},
        "worker-service":         {"status": "unhealthy", "latency_ms": None, "error_rate": 1.0},
    }

    data = health_data.get(
        service_name.lower(),
        {"status": "unknown", "latency_ms": None, "error_rate": None}
    )

    return json.dumps({
        "service": service_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        **data,
    })


@mcp.tool()
def list_recent_incidents(limit: int = 5) -> str:
    """
    List the most recent incidents from historical memory.

    Args:
        limit: Maximum number of incidents to return
    """
    try:
        from memory.historical_memory import HistoricalMemory
        mem = HistoricalMemory()
        recent = mem.get_recent(limit=limit)
        return json.dumps({
            "status": "ok",
            "count": len(recent),
            "incidents": recent,
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def get_platform_stats() -> str:
    """
    Get operational statistics for the incident resolution platform.
    Returns model config, knowledge base size, and incident count.
    """
    try:
        from knowledge_base.retriever import get_collection_stats
        from memory.historical_memory import HistoricalMemory
        from config.settings import settings

        kb_stats = get_collection_stats()
        mem = HistoricalMemory()

        return json.dumps({
            "status": "ok",
            "models": {
                "coordinator": settings.coordinator_model,
                "knowledge":   settings.knowledge_model,
                "resolution":  settings.resolution_model,
                "embeddings":  settings.embedding_model,
            },
            "knowledge_base": kb_stats,
            "historical_incidents": mem.count(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


if __name__ == "__main__":
    import asyncio
    logger.info(
        f"Starting MCP server on "
        f"{settings.mcp_server_host}:{settings.mcp_server_port}"
    )
    mcp.run(
        transport="stdio",  # stdio for local, sse for network
    )