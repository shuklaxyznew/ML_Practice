"""
agents/job_matching_agent.py
──────────────────────────────
Job Matching Agent — scores and ranks jobs against the candidate's profile.
"""

from __future__ import annotations

from crewai import Agent

from config import settings
from tools.matching_tools import compute_match_score, identify_skill_gaps, rank_jobs_by_match
from utils.llm_factory import get_llm


def create_job_matching_agent() -> Agent:
    return Agent(
        role="AI Job Matching Specialist",
        goal=(
            "Analyse semantic similarity between resumes and job descriptions. "
            "Score every job from 0-100, explain why each matches, "
            "identify missing skills, and rank jobs by relevance."
        ),
        backstory=(
            "You are a pioneer in AI-powered recruitment, having built matching algorithms at top ATS companies. "
            "You think in vectors and embeddings, but communicate in plain English. "
            "You are honest: you never oversell a weak match. You give candidates clear, actionable "
            "feedback on exactly why they do or don't fit a role, and what to do about it."
        ),
        tools=[compute_match_score, rank_jobs_by_match, identify_skill_gaps],
        llm=get_llm(),
        verbose=settings.crew_verbose,
        max_iter=settings.crew_max_iter,
        memory=True,
        allow_delegation=False,
    )
