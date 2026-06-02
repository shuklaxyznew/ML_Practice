"""
agents/application_tracking_agent.py
──────────────────────────────────────
Application Tracking Agent — manages the full application lifecycle in the database.
"""

from __future__ import annotations

import json
from typing import Any

from crewai import Agent
from crewai.tools import tool
from loguru import logger

from config import settings
from database.connection import get_session
from database.models import Application, ApplicationStatus, Job, MatchScore, Resume
from database.repository import ApplicationRepository, JobRepository, MatchScoreRepository
from utils.llm_factory import get_llm


@tool("save_jobs_to_database")
def save_jobs_to_database(jobs_json: str) -> str:
    """
    Persist a list of job listings to the database.
    Deduplicates by external_id.

    Args:
        jobs_json: JSON array of job dictionaries.

    Returns:
        JSON with saved_count and skipped_count.
    """
    import asyncio

    async def _save():
        jobs = json.loads(jobs_json)
        saved = 0
        skipped = 0
        async with get_session() as session:
            repo = JobRepository(session)
            for job_data in jobs:
                ext_id = job_data.get("external_id", "")
                if ext_id and await repo.get_by_external_id(ext_id):
                    skipped += 1
                    continue
                job = Job(
                    title=job_data.get("title", "Unknown"),
                    company=job_data.get("company", "Unknown"),
                    location=job_data.get("location"),
                    is_remote=job_data.get("is_remote", False),
                    description=job_data.get("description"),
                    salary_min=job_data.get("salary_min"),
                    salary_max=job_data.get("salary_max"),
                    required_skills=job_data.get("required_skills", []),
                    application_url=job_data.get("application_url"),
                    source=job_data.get("source", "other"),
                    external_id=ext_id,
                    raw_data=job_data,
                )
                await repo.create(job)
                saved += 1
        return {"saved": saved, "skipped": skipped, "total": saved + skipped}

    result = asyncio.run(_save())
    return json.dumps(result)


@tool("create_application")
def create_application(job_id: str, resume_id: str, cover_letter: str = "") -> str:
    """
    Create a new application record with APPLIED status.

    Args:
        job_id:       Database ID of the job being applied to.
        resume_id:    Database ID of the resume being used.
        cover_letter: Cover letter text.

    Returns:
        JSON with application_id and status.
    """
    import asyncio
    from datetime import datetime

    async def _create():
        async with get_session() as session:
            repo = ApplicationRepository(session)
            app = Application(
                job_id=job_id,
                resume_id=resume_id,
                status=ApplicationStatus.APPLIED,
                applied_at=datetime.utcnow(),
                cover_letter=cover_letter,
            )
            saved = await repo.create(app)
            return {"application_id": saved.id, "status": saved.status}

    result = asyncio.run(_create())
    return json.dumps(result)


@tool("update_application_status")
def update_application_status(application_id: str, new_status: str, notes: str = "") -> str:
    """
    Update an application's status (e.g., APPLIED → INTERVIEW → OFFER).

    Args:
        application_id: Database ID of the application.
        new_status:     New status: applied|phone_screen|interview|final_round|offer|rejected|withdrawn
        notes:          Optional notes about the status change.

    Returns:
        JSON with updated status.
    """
    import asyncio

    async def _update():
        async with get_session() as session:
            repo = ApplicationRepository(session)
            try:
                status_enum = ApplicationStatus(new_status.lower())
            except ValueError:
                return {"error": f"Invalid status: {new_status}"}
            await repo.update_status(application_id, status_enum, notes=notes)
            return {"application_id": application_id, "new_status": new_status, "updated": True}

    result = asyncio.run(_update())
    return json.dumps(result)


@tool("get_application_statistics")
def get_application_statistics() -> str:
    """
    Retrieve aggregate application statistics for reporting.

    Returns:
        JSON with counts by status, total jobs, and weekly trends.
    """
    import asyncio

    async def _stats():
        async with get_session() as session:
            app_repo = ApplicationRepository(session)
            job_repo = JobRepository(session)

            status_counts = await app_repo.status_counts()
            source_counts = await job_repo.count_by_source()
            total_jobs = sum(source_counts.values())
            total_apps = sum(status_counts.values())

            return {
                "applications_by_status": status_counts,
                "jobs_by_source": source_counts,
                "total_jobs_discovered": total_jobs,
                "total_applications": total_apps,
                "response_rate": (
                    round(
                        (status_counts.get("interview", 0) + status_counts.get("phone_screen", 0))
                        / max(total_apps, 1)
                        * 100,
                        1,
                    )
                ),
                "offer_rate": round(
                    status_counts.get("offer", 0) / max(total_apps, 1) * 100, 1
                ),
            }

    result = asyncio.run(_stats())
    return json.dumps(result)


@tool("export_applications_csv")
def export_applications_csv(output_path: str = "./data/reports/applications.csv") -> str:
    """
    Export all applications to a CSV file.

    Args:
        output_path: Where to save the CSV file.

    Returns:
        JSON with file_path and row_count.
    """
    import asyncio
    from pathlib import Path

    import pandas as pd
    from sqlalchemy import select, join

    async def _export():
        async with get_session() as session:
            result = await session.execute(
                select(
                    Application.id,
                    Application.status,
                    Application.applied_at,
                    Application.notes,
                    Job.title,
                    Job.company,
                    Job.location,
                    Job.application_url,
                    Job.salary_min,
                    Job.salary_max,
                ).select_from(
                    Application.__table__.join(Job.__table__, Application.job_id == Job.id)
                )
            )
            rows = result.fetchall()
            return [dict(r._mapping) for r in rows]

    data = asyncio.run(_export())
    if not data:
        return json.dumps({"error": "No applications found", "row_count": 0})

    df = pd.DataFrame(data)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out), index=False)
    logger.info(f"Exported {len(df)} applications to {output_path}")
    return json.dumps({"file_path": str(out), "row_count": len(df)})


def create_application_tracking_agent() -> Agent:
    return Agent(
        role="Application Tracker & Data Manager",
        goal=(
            "Maintain a complete, accurate database of all job applications. "
            "Track every status change from discovery through offer/rejection. "
            "Generate CSV reports and provide analytics on application pipeline health."
        ),
        backstory=(
            "You are a systematic operations specialist with a background in CRM systems and "
            "data management. Nothing slips through your cracks. "
            "You maintain the single source of truth for the entire job search pipeline, "
            "and your reports give candidates actionable insight into their job search strategy."
        ),
        tools=[
            save_jobs_to_database,
            create_application,
            update_application_status,
            get_application_statistics,
            export_applications_csv,
        ],
        llm=get_llm(),
        verbose=settings.crew_verbose,
        max_iter=settings.crew_max_iter,
        memory=True,
        allow_delegation=False,
    )
