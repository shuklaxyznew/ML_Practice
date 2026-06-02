"""
agents/resume_analysis_agent.py
─────────────────────────────────
Resume Analysis Agent — parses, structures, and embeds resumes.
"""

from __future__ import annotations

from crewai import Agent

from config import settings
from tools.resume_tools import embed_resume, parse_resume
from utils.llm_factory import get_llm


def create_resume_analysis_agent() -> Agent:
    return Agent(
        role="Expert Resume Analyst",
        goal=(
            "Parse PDF/DOCX resumes into structured data. "
            "Extract skills, experience, education, and projects with high accuracy. "
            "Create semantic embeddings and store them for matching."
        ),
        backstory=(
            "You are a former technical recruiter turned AI specialist. "
            "You have reviewed over 50,000 resumes across software engineering, AI/ML, and data roles. "
            "You understand what makes a resume stand out: quantified achievements, relevant keywords, "
            "and clear narrative. You excel at extracting signal from noise in unstructured resume text."
        ),
        tools=[parse_resume, embed_resume],
        llm=get_llm(),
        verbose=settings.crew_verbose,
        max_iter=settings.crew_max_iter,
        memory=True,
        allow_delegation=False,
    )
