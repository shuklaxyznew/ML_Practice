"""
agents/job_research_agent.py
─────────────────────────────
Job Research Agent — searches multiple platforms and deduplicates results.
"""

from __future__ import annotations

from crewai import Agent

from config import settings
from tools.scraping_tools import (
    scrape_company_jobs,
    search_indeed_jobs,
    search_linkedin_jobs,
    search_wellfound_jobs,
)
from utils.llm_factory import get_llm


def create_job_research_agent() -> Agent:
    return Agent(
        role="Senior Job Research Specialist",
        goal=(
            "Search job listings from LinkedIn, Indeed, Wellfound, and company career pages. "
            "Extract structured job data, remove duplicates, and return a clean consolidated list."
        ),
        backstory=(
            "You are a meticulous research specialist with 10 years of experience in talent acquisition. "
            "You know every corner of the job market — from startup job boards to FAANG career pages. "
            "You are obsessive about data quality: no duplicates, no stale listings, no missing fields. "
            "You always validate that salary, skills, and application links are captured."
        ),
        tools=[
            search_linkedin_jobs,
            search_indeed_jobs,
            search_wellfound_jobs,
            scrape_company_jobs,
        ],
        llm=get_llm(),
        verbose=settings.crew_verbose,
        max_iter=settings.crew_max_iter,
        memory=True,
        allow_delegation=False,
    )
