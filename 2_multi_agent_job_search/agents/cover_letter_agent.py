"""
agents/cover_letter_agent.py
──────────────────────────────
Cover Letter Agent — generates personalised, company-specific cover letters.
"""

from __future__ import annotations

from crewai import Agent

from config import settings
from tools.writing_tools import generate_cover_letter, generate_interview_prep, research_company
from utils.llm_factory import get_llm


def create_cover_letter_agent() -> Agent:
    return Agent(
        role="Professional Cover Letter Writer",
        goal=(
            "Generate compelling, personalised cover letters for each job application. "
            "Research the company context, connect candidate achievements to company needs, "
            "and write in a voice that is professional yet distinctly human."
        ),
        backstory=(
            "You are a professional ghostwriter who has helped 2,000+ engineers land roles at "
            "top-tier companies. You hate generic cover letters as much as recruiters do. "
            "You obsessively research every company before writing a single word. "
            "Your letters always open with a hook, tell a story, and close with conviction. "
            "You also prepare candidates for interviews with targeted questions and talking points."
        ),
        tools=[generate_cover_letter, research_company, generate_interview_prep],
        llm=get_llm(temperature=0.4),
        verbose=settings.crew_verbose,
        max_iter=settings.crew_max_iter,
        memory=True,
        allow_delegation=False,
    )
