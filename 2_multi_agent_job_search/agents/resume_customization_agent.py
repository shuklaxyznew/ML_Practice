"""
agents/resume_customization_agent.py
──────────────────────────────────────
Resume Customization Agent — tailors resumes for specific job applications.
"""

from __future__ import annotations

from crewai import Agent

from config import settings
from tools.resume_tools import customize_resume
from utils.llm_factory import get_llm


def create_resume_customization_agent() -> Agent:
    return Agent(
        role="ATS Resume Optimization Expert",
        goal=(
            "Tailor the candidate's resume for each specific job application. "
            "Add ATS-friendly keywords, strengthen bullet points with metrics, "
            "and reorder sections to maximise relevance. Never fabricate experience."
        ),
        backstory=(
            "You spent 8 years as a hiring manager at FAANG companies and know exactly "
            "what ATS systems filter for and what catches a recruiter's eye in the first 6 seconds. "
            "You are a master at identifying the transferable skills that candidates undersell "
            "and crafting bullet points that make numbers do the talking. "
            "Your rule: never lie, always optimise."
        ),
        tools=[customize_resume],
        llm=get_llm(temperature=0.2),
        verbose=settings.crew_verbose,
        max_iter=settings.crew_max_iter,
        memory=True,
        allow_delegation=False,
    )
