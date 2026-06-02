"""
database/models.py
──────────────────
SQLAlchemy 2.x ORM models for the entire job search system.
All tables use UUID primary keys and include created_at / updated_at audit columns.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    Enum as SAEnum,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
import enum


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.utcnow()


class Base(DeclarativeBase):
    pass


# ── Enums ────────────────────────────────────────────────────


class ApplicationStatus(str, enum.Enum):
    DISCOVERED = "discovered"
    APPLIED = "applied"
    PHONE_SCREEN = "phone_screen"
    INTERVIEW = "interview"
    FINAL_ROUND = "final_round"
    OFFER = "offer"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


class ExperienceLevel(str, enum.Enum):
    INTERN = "intern"
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"
    STAFF = "staff"
    PRINCIPAL = "principal"
    DIRECTOR = "director"


class JobSource(str, enum.Enum):
    LINKEDIN = "linkedin"
    INDEED = "indeed"
    WELLFOUND = "wellfound"
    GLASSDOOR = "glassdoor"
    COMPANY_SITE = "company_site"
    OTHER = "other"


# ── Models ───────────────────────────────────────────────────


class Resume(Base):
    """Uploaded and parsed resume."""

    __tablename__ = "resumes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String(255))
    file_path: Mapped[str] = mapped_column(String(512))
    raw_text: Mapped[Optional[str]] = mapped_column(Text)
    parsed_data: Mapped[Optional[dict]] = mapped_column(JSON)  # skills, education, exp
    embedding_id: Mapped[Optional[str]] = mapped_column(String(255))  # FAISS/Chroma ref
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)

    # Relationships
    applications: Mapped[list["Application"]] = relationship(back_populates="resume")
    match_scores: Mapped[list["MatchScore"]] = relationship(back_populates="resume")


class Job(Base):
    """Job listing discovered by the Job Research Agent."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    external_id: Mapped[Optional[str]] = mapped_column(String(255), index=True)
    title: Mapped[str] = mapped_column(String(255))
    company: Mapped[str] = mapped_column(String(255))
    location: Mapped[Optional[str]] = mapped_column(String(255))
    is_remote: Mapped[bool] = mapped_column(Boolean, default=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    requirements: Mapped[Optional[str]] = mapped_column(Text)
    salary_min: Mapped[Optional[float]] = mapped_column(Float)
    salary_max: Mapped[Optional[float]] = mapped_column(Float)
    salary_currency: Mapped[str] = mapped_column(String(10), default="USD")
    experience_level: Mapped[Optional[str]] = mapped_column(
        SAEnum(ExperienceLevel), nullable=True
    )
    required_skills: Mapped[Optional[list]] = mapped_column(JSON)
    nice_to_have_skills: Mapped[Optional[list]] = mapped_column(JSON)
    application_url: Mapped[Optional[str]] = mapped_column(String(1024))
    source: Mapped[str] = mapped_column(SAEnum(JobSource), default=JobSource.OTHER)
    company_description: Mapped[Optional[str]] = mapped_column(Text)
    company_size: Mapped[Optional[str]] = mapped_column(String(100))
    industry: Mapped[Optional[str]] = mapped_column(String(255))
    posted_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    raw_data: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)

    # Relationships
    applications: Mapped[list["Application"]] = relationship(back_populates="job")
    match_scores: Mapped[list["MatchScore"]] = relationship(back_populates="job")


class MatchScore(Base):
    """Semantic match between a resume and a job."""

    __tablename__ = "match_scores"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    resume_id: Mapped[str] = mapped_column(ForeignKey("resumes.id"), index=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
    score: Mapped[float] = mapped_column(Float)          # 0–100
    explanation: Mapped[Optional[str]] = mapped_column(Text)
    missing_skills: Mapped[Optional[list]] = mapped_column(JSON)
    matching_skills: Mapped[Optional[list]] = mapped_column(JSON)
    recommendation: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    resume: Mapped["Resume"] = relationship(back_populates="match_scores")
    job: Mapped["Job"] = relationship(back_populates="match_scores")


class Application(Base):
    """Job application lifecycle tracking."""

    __tablename__ = "applications"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
    resume_id: Mapped[str] = mapped_column(ForeignKey("resumes.id"), index=True)
    status: Mapped[str] = mapped_column(
        SAEnum(ApplicationStatus), default=ApplicationStatus.DISCOVERED
    )
    applied_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    cover_letter: Mapped[Optional[str]] = mapped_column(Text)
    customized_resume_path: Mapped[Optional[str]] = mapped_column(String(512))
    notes: Mapped[Optional[str]] = mapped_column(Text)
    follow_up_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    interview_dates: Mapped[Optional[list]] = mapped_column(JSON)
    offer_amount: Mapped[Optional[float]] = mapped_column(Float)
    rejection_reason: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)

    job: Mapped["Job"] = relationship(back_populates="applications")
    resume: Mapped["Resume"] = relationship(back_populates="applications")


class Notification(Base):
    """Email/alert log."""

    __tablename__ = "notifications"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    type: Mapped[str] = mapped_column(String(100))       # "weekly_report", "new_jobs", etc.
    recipient: Mapped[str] = mapped_column(String(255))
    subject: Mapped[str] = mapped_column(String(512))
    body: Mapped[Optional[str]] = mapped_column(Text)
    sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    is_sent: Mapped[bool] = mapped_column(Boolean, default=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class SearchHistory(Base):
    """Audit log of every job search run."""

    __tablename__ = "search_history"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    keywords: Mapped[str] = mapped_column(String(512))
    location: Mapped[Optional[str]] = mapped_column(String(255))
    sources: Mapped[Optional[list]] = mapped_column(JSON)
    jobs_found: Mapped[int] = mapped_column(Integer, default=0)
    jobs_matched: Mapped[int] = mapped_column(Integer, default=0)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(50), default="completed")
    error: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
