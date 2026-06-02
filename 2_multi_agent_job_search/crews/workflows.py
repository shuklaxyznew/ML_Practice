"""
crews/workflows.py
───────────────────
CrewAI Crew definitions and orchestration.

Three workflows:
1. FullJobSearchCrew    — end-to-end: search → analyse → match → apply
2. QuickMatchCrew       — match existing jobs to a newly uploaded resume
3. ApplicationCrew      — generate cover letter + customised resume for a specific job
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from crewai import Crew, Process
from loguru import logger

from agents import (
    create_application_tracking_agent,
    create_cover_letter_agent,
    create_job_matching_agent,
    create_job_research_agent,
    create_notification_agent,
    create_resume_analysis_agent,
    create_resume_customization_agent,
)
from config import settings
from tasks.job_search_tasks import (
    analyse_resume_task,
    customize_resume_task,
    match_jobs_task,
    research_jobs_task,
    send_notifications_task,
    track_applications_task,
    write_cover_letter_task,
)


@dataclass
class SearchConfig:
    """Configuration for a job search run."""

    keywords: str
    location: str = "Remote"
    resume_path: str = ""
    target_job_title: str = ""
    target_company: str = ""
    min_match_score: float = 60.0


@dataclass
class CrewResult:
    """Structured result from a Crew run."""

    raw_output: str
    jobs_found: int = 0
    top_matches: list[dict[str, Any]] = field(default_factory=list)
    applications_created: int = 0
    error: str | None = None


# ── Full End-to-End Crew ──────────────────────────────────────


class FullJobSearchCrew:
    """
    Sequential workflow:
    Research → Analyse Resume → Match Jobs → Customise Resume →
    Write Cover Letter → Track Applications → Send Notifications
    """

    def __init__(self, config: SearchConfig) -> None:
        self.config = config
        self._agents = self._create_agents()

    def _create_agents(self) -> dict:
        return {
            "researcher": create_job_research_agent(),
            "analyser": create_resume_analysis_agent(),
            "matcher": create_job_matching_agent(),
            "customiser": create_resume_customization_agent(),
            "writer": create_cover_letter_agent(),
            "tracker": create_application_tracking_agent(),
            "notifier": create_notification_agent(),
        }

    def run(self) -> CrewResult:
        cfg = self.config
        a = self._agents

        # Build tasks with context dependencies
        t_research = research_jobs_task(a["researcher"], cfg.keywords, cfg.location)
        t_analyse = analyse_resume_task(a["analyser"], cfg.resume_path)
        t_match = match_jobs_task(a["matcher"], [t_research, t_analyse])
        t_customise = customize_resume_task(
            a["customiser"],
            cfg.target_job_title or cfg.keywords,
            cfg.target_company or "Top Matches",
            [t_analyse, t_match],
        )
        t_cover = write_cover_letter_task(
            a["writer"],
            cfg.target_job_title or cfg.keywords,
            cfg.target_company or "Top Match Company",
            [t_analyse, t_match],
        )
        t_track = track_applications_task(a["tracker"], "[]", [t_research, t_match])
        t_notify = send_notifications_task(a["notifier"], [t_match, t_track])

        crew = Crew(
            agents=list(self._agents.values()),
            tasks=[t_research, t_analyse, t_match, t_customise, t_cover, t_track, t_notify],
            process=Process.sequential,
            verbose=settings.crew_verbose,
            memory=settings.crew_memory,
            max_rpm=10,  # Rate limit: 10 LLM calls/min
        )

        logger.info(f"Starting FullJobSearchCrew: keywords={cfg.keywords}, location={cfg.location}")
        try:
            result = crew.kickoff()
            return CrewResult(raw_output=str(result))
        except Exception as e:
            logger.error(f"FullJobSearchCrew failed: {e}")
            return CrewResult(raw_output="", error=str(e))


# ── Quick Match Crew (no scraping) ────────────────────────────


class QuickMatchCrew:
    """
    Lightweight workflow for matching a new resume against already-saved jobs.
    Sequential: Analyse Resume → Match Jobs
    """

    def __init__(self, resume_path: str, keywords: str = "") -> None:
        self.resume_path = resume_path
        self.keywords = keywords

    def run(self) -> CrewResult:
        analyser = create_resume_analysis_agent()
        matcher = create_job_matching_agent()

        t_analyse = analyse_resume_task(analyser, self.resume_path)
        t_match = match_jobs_task(matcher, [t_analyse])

        crew = Crew(
            agents=[analyser, matcher],
            tasks=[t_analyse, t_match],
            process=Process.sequential,
            verbose=settings.crew_verbose,
            memory=False,
        )

        logger.info(f"Starting QuickMatchCrew: {self.resume_path}")
        try:
            result = crew.kickoff()
            return CrewResult(raw_output=str(result))
        except Exception as e:
            logger.error(f"QuickMatchCrew failed: {e}")
            return CrewResult(raw_output="", error=str(e))


# ── Application Crew (single job) ────────────────────────────


class ApplicationCrew:
    """
    Generate a tailored resume + cover letter for a single job.
    Sequential: Customise Resume → Write Cover Letter → Track
    """

    def __init__(
        self,
        resume_text: str,
        job_title: str,
        job_description: str,
        company_name: str,
    ) -> None:
        self.resume_text = resume_text
        self.job_title = job_title
        self.job_description = job_description
        self.company_name = company_name

    def run(self) -> CrewResult:
        customiser = create_resume_customization_agent()
        writer = create_cover_letter_agent()
        tracker = create_application_tracking_agent()

        # Inject data directly into task descriptions for single-job flow
        t_customise = customize_resume_task(customiser, self.job_title, self.company_name, [])
        t_cover = write_cover_letter_task(writer, self.job_title, self.company_name, [t_customise])

        crew = Crew(
            agents=[customiser, writer],
            tasks=[t_customise, t_cover],
            process=Process.sequential,
            verbose=settings.crew_verbose,
            memory=False,
        )

        logger.info(f"Starting ApplicationCrew: {self.job_title} @ {self.company_name}")
        try:
            result = crew.kickoff()
            return CrewResult(raw_output=str(result))
        except Exception as e:
            logger.error(f"ApplicationCrew failed: {e}")
            return CrewResult(raw_output="", error=str(e))
