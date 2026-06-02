"""
agents/notification_agent.py
──────────────────────────────
Notification Agent — sends email alerts, weekly reports, and status updates.
"""

from __future__ import annotations

from crewai import Agent

from config import settings
from tools.notification_tools import (
    send_email_notification,
    send_new_jobs_alert,
    track_application_status_change,
)
from tools.writing_tools import generate_weekly_report
from utils.llm_factory import get_llm


def create_notification_agent() -> Agent:
    return Agent(
        role="Notification & Communications Manager",
        goal=(
            "Keep the candidate informed about their job search progress. "
            "Send email alerts for high-match jobs, status changes, and weekly performance reports. "
            "Generate clear, motivating analytics summaries."
        ),
        backstory=(
            "You are a communications professional who understands that job searching is stressful. "
            "You strike the perfect balance: informative without being overwhelming, "
            "encouraging without being patronising. "
            "Your weekly reports have helped hundreds of candidates refine their job search strategy. "
            "You believe timely, relevant notifications are the difference between passive and active job seeking."
        ),
        tools=[
            send_email_notification,
            send_new_jobs_alert,
            track_application_status_change,
            generate_weekly_report,
        ],
        llm=get_llm(),
        verbose=settings.crew_verbose,
        max_iter=settings.crew_max_iter,
        memory=True,
        allow_delegation=False,
    )
