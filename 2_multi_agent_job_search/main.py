"""
main.py
────────
CLI entry point for the Multi-Agent Job Search Assistant.

Usage:
    python main.py search --keywords "ML Engineer" --location Remote
    python main.py match --resume path/to/resume.pdf
    python main.py apply --job-id <id> --resume path/to/resume.pdf
    python main.py init-db
    python main.py dashboard
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import click
from loguru import logger

from config import settings
from utils.logger import setup_logging


@click.group()
def cli():
    """Multi-Agent Job Search Assistant CLI."""
    setup_logging()


@cli.command()
def init_db():
    """Initialise the database (create all tables)."""
    from database.connection import init_db as _init_db

    asyncio.run(_init_db())
    click.echo("✅ Database initialised.")


@cli.command()
@click.option("--keywords", "-k", required=True, help="Job search keywords (comma-separated)")
@click.option("--location", "-l", default="Remote", help="Job location")
@click.option("--resume", "-r", default="", help="Path to resume PDF/DOCX")
@click.option("--min-score", default=60.0, help="Minimum match score (0-100)")
def search(keywords: str, location: str, resume: str, min_score: float):
    """Run the full end-to-end job search crew."""
    from crews.workflows import FullJobSearchCrew, SearchConfig

    config = SearchConfig(
        keywords=keywords,
        location=location,
        resume_path=resume,
        min_match_score=min_score,
    )
    click.echo(f"🚀 Launching job search: {keywords} in {location}")
    crew = FullJobSearchCrew(config)
    result = crew.run()

    if result.error:
        click.echo(f"❌ Error: {result.error}", err=True)
        sys.exit(1)
    else:
        click.echo("✅ Job search complete!")
        click.echo(result.raw_output[:500])


@cli.command()
@click.option("--resume", "-r", required=True, help="Path to resume PDF/DOCX")
def match(resume: str):
    """Match a resume against all saved jobs in the database."""
    from crews.workflows import QuickMatchCrew

    if not Path(resume).exists():
        click.echo(f"❌ Resume file not found: {resume}", err=True)
        sys.exit(1)

    click.echo(f"🧠 Matching resume: {resume}")
    crew = QuickMatchCrew(resume_path=resume)
    result = crew.run()

    if result.error:
        click.echo(f"❌ Error: {result.error}", err=True)
        sys.exit(1)
    else:
        click.echo("✅ Matching complete!")
        click.echo(result.raw_output[:500])


@cli.command()
@click.option("--resume", "-r", required=True, help="Path to resume PDF/DOCX")
@click.option("--job-title", required=True, help="Job title")
@click.option("--company", required=True, help="Company name")
@click.option("--job-desc", default="", help="Job description text or file path")
def apply(resume: str, job_title: str, company: str, job_desc: str):
    """Generate customised resume + cover letter for a specific job."""
    from crews.workflows import ApplicationCrew

    # Load job description from file if path provided
    if Path(job_desc).exists():
        job_description = Path(job_desc).read_text()
    else:
        job_description = job_desc

    with open(resume) as f:
        resume_text = f.read()

    crew = ApplicationCrew(
        resume_text=resume_text,
        job_title=job_title,
        job_description=job_description,
        company_name=company,
    )
    click.echo(f"✍️ Generating application materials for {job_title} @ {company}")
    result = crew.run()

    if result.error:
        click.echo(f"❌ Error: {result.error}", err=True)
        sys.exit(1)
    else:
        click.echo("✅ Application materials generated!")
        click.echo(result.raw_output[:800])


@cli.command()
def dashboard():
    """Launch the Streamlit dashboard."""
    click.echo("🖥️ Launching Streamlit dashboard...")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "frontend/app.py"],
        check=True,
    )


@cli.command()
def report():
    """Generate and send a weekly job search report."""
    click.echo("📊 Generating weekly report...")
    from tools.writing_tools import generate_weekly_report
    from tools.notification_tools import send_email_notification
    from agents.application_tracking_agent import get_application_statistics

    stats = get_application_statistics()
    report_json = generate_weekly_report(stats)

    import json
    report_data = json.loads(report_json)
    click.echo(report_data.get("report", "Report generated."))


if __name__ == "__main__":
    cli()
