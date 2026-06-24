import os
import warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"
warnings.filterwarnings("ignore", message="Failed to send telemetry")

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import uuid
import time
from observability.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)

app = FastAPI(
    title="Enterprise Incident Resolution Agent",
    description="Multi-agent AI platform for autonomous incident investigation",
    version="1.0.0",
)

# In-memory job store — Phase 8 replaces with Redis
_jobs: dict = {}


# ── Request / Response Models ──

class IncidentRequest(BaseModel):
    title: str = Field(description="Short incident title")
    description: str = Field(description="Full incident description")
    affected_service: str = Field(description="Name of the affected service")
    severity: Optional[str] = Field(default="unknown", description="P1/P2/P3/P4")
    raw_logs: Optional[str] = Field(default="", description="Raw log text")
    incident_id: Optional[str] = Field(default=None, description="Optional custom ID")


class IncidentResponse(BaseModel):
    job_id: str
    incident_id: str
    status: str
    message: str
    submitted_at: float


class JobStatusResponse(BaseModel):
    job_id: str
    incident_id: str
    status: str
    submitted_at: float
    completed_at: Optional[float]
    duration_seconds: Optional[float]
    result: Optional[dict]
    error: Optional[str]


# ── Background Worker ──

def investigate_in_background(job_id: str, incident_data: dict) -> None:
    """
    Runs the full LangGraph investigation pipeline in the background.
    Updates job store with results when complete.
    """
    _jobs[job_id]["status"] = "running"
    logger.info(f"Background investigation started: {job_id}")

    MAX_INVESTIGATION_SECONDS = 600  # 10 minute hard timeout

    _jobs[job_id]["status"] = "running"
    start = time.time()

    try:
        from main1crew import run_incident
        final_state = run_incident(
            incident_id=incident_data["incident_id"],
            title=incident_data["title"],
            description=incident_data["description"],
            affected_service=incident_data["affected_service"],
            severity=incident_data["severity"],
            raw_logs=incident_data.get("raw_logs", ""),
            run_evaluation=True,
        )

        _jobs[job_id].update({
            "status":       "complete",
            "completed_at": time.time(),
            "duration_seconds": round(
                time.time() - _jobs[job_id]["submitted_at"], 2
            ),
            "result": {
                "final_report":     final_state.get("final_report", ""),
                "root_cause":       final_state.get("root_cause", ""),
                "confidence_score": final_state.get("confidence_score", 0),
                "recommendations":  final_state.get("recommendations", []),
                "tool_calls_made":  final_state.get("tool_calls_made", []),
                "evaluation":       final_state.get("evaluation", {}),
            },
        })
        logger.info(f"Investigation complete: {job_id}")

        # Check timeout inside the function
        if time.time() - start > MAX_INVESTIGATION_SECONDS:
            raise TimeoutError("Investigation exceeded 10 minute limit")

    except Exception as e:
        logger.error(f"Investigation failed: {job_id} — {e}")
        _jobs[job_id].update({
            "status":       "failed",
            "completed_at": time.time(),
            "duration_seconds": round(
                time.time() - _jobs[job_id]["submitted_at"], 2
            ),
            "error": str(e),
        })


# ── Endpoints ──

@app.get("/health")
def health_check():
    """
    Health check endpoint.
    Kubernetes liveness probe hits this every 30 seconds.
    Returns 200 if platform is ready, 503 if not.
    """
    try:
        # Check Ollama is reachable
        import ollama
        models = ollama.list()
        model_names = [m.model for m in models.models]

        coordinator_ok = settings.coordinator_model in model_names
        resolution_ok = settings.resolution_model in model_names

        if not coordinator_ok or not resolution_ok:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "degraded",
                    "reason": "Required models not available",
                    "available_models": model_names,
                }
            )

        return {
            "status":   "healthy",
            "models":   model_names,
            "jobs":     len(_jobs),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": str(e)}
        )


@app.post("/incidents", response_model=IncidentResponse)
async def submit_incident(
    request: IncidentRequest,
    background_tasks: BackgroundTasks,
):
    """
    Submit an incident for autonomous investigation.
    Investigation runs in the background.
    Poll /incidents/{job_id} for results.

    This endpoint is called by:
    - PagerDuty webhooks
    - Prometheus alertmanager
    - Grafana alerts
    - Manual curl commands
    - Your monitoring systems
    """
    incident_id = request.incident_id or f"INC-{uuid.uuid4().hex[:8].upper()}"
    job_id = f"JOB-{uuid.uuid4().hex[:8].upper()}"
    submitted_at = time.time()

    incident_data = {
        "incident_id":      incident_id,
        "title":            request.title,
        "description":      request.description,
        "affected_service": request.affected_service,
        "severity":         request.severity or "unknown",
        "raw_logs":         request.raw_logs or "",
    }

    _jobs[job_id] = {
        "job_id":       job_id,
        "incident_id":  incident_id,
        "status":       "queued",
        "submitted_at": submitted_at,
        "completed_at": None,
        "duration_seconds": None,
        "result":       None,
        "error":        None,
        "incident_data": incident_data,
    }

    background_tasks.add_task(
        investigate_in_background, job_id, incident_data
    )

    logger.info(f"Incident queued: {incident_id} → job: {job_id}")

    return IncidentResponse(
        job_id=job_id,
        incident_id=incident_id,
        status="queued",
        message="Investigation started. Poll /incidents/{job_id} for results.",
        submitted_at=submitted_at,
    )


@app.get("/incidents/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    """
    Get the status and results of an incident investigation.
    Status values: queued → running → complete | failed
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    return JobStatusResponse(**{k: v for k, v in job.items()
                                if k != "incident_data"})


@app.get("/incidents")
def list_jobs(limit: int = 10):
    """List recent investigation jobs."""
    jobs = sorted(
        _jobs.values(),
        key=lambda j: j["submitted_at"],
        reverse=True,
    )[:limit]
    return {
        "total": len(_jobs),
        "jobs": [
            {
                "job_id":      j["job_id"],
                "incident_id": j["incident_id"],
                "status":      j["status"],
                "submitted_at": j["submitted_at"],
            }
            for j in jobs
        ],
    }


@app.delete("/incidents/{job_id}")
def delete_job(job_id: str):
    """Remove a job from the store."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    del _jobs[job_id]
    return {"deleted": job_id}