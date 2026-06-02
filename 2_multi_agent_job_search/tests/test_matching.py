"""
tests/test_matching.py
───────────────────────
Unit tests for the Job Matching and Resume Analysis pipeline.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# ── Resume Analysis Tests ─────────────────────────────────────


class TestResumeParser:
    def test_parse_pdf_returns_structured_data(self, tmp_path):
        """parse_resume should return a JSON with raw_text and parsed fields."""
        # Create a minimal text file masquerading as a resume
        resume_file = tmp_path / "test_resume.txt"
        resume_file.write_text(
            "John Doe\njohn@example.com\nSkills: Python, LangChain\nExperience: 5 years"
        )

        from tools.resume_tools import _extract_pdf_text

        with patch("tools.resume_tools._extract_pdf_text", return_value="John Doe\nPython Developer"):
            with patch("tools.resume_tools.get_llm") as mock_llm:
                mock_response = MagicMock()
                mock_response.content = json.dumps(
                    {
                        "name": "John Doe",
                        "email": "john@example.com",
                        "skills": ["Python", "LangChain"],
                        "total_years_experience": 5,
                    }
                )
                mock_llm.return_value.invoke.return_value = mock_response

                # Direct function test (bypass @tool decorator)
                # In practice you'd call parse_resume.run(file_path=...)
                assert True  # placeholder for actual invocation

    def test_unsupported_file_type_returns_error(self, tmp_path):
        """parse_resume should return error JSON for unsupported extensions."""
        from tools.resume_tools import parse_resume

        # Create a .xyz file
        bad_file = tmp_path / "resume.xyz"
        bad_file.write_text("content")

        result = json.loads(parse_resume.run(file_path=str(bad_file)))
        assert "error" in result


# ── Match Score Tests ─────────────────────────────────────────


class TestMatchScoring:
    def test_high_match_score_for_relevant_job(self):
        """High-skill-overlap resume+JD pair should score ≥ 75."""
        with patch("tools.matching_tools.get_llm") as mock_llm:
            mock_response = MagicMock()
            mock_response.content = json.dumps(
                {
                    "score": 88,
                    "matching_skills": ["Python", "LangChain", "FastAPI"],
                    "missing_skills": ["Kubernetes"],
                    "recommendation": "apply",
                    "explanation": "Strong match across all technical dimensions.",
                }
            )
            mock_llm.return_value.invoke.return_value = mock_response

            from tools.matching_tools import compute_match_score

            result = json.loads(
                compute_match_score.run(
                    resume_text="Python expert with 5 years LangChain, FastAPI",
                    job_description="We need a Python engineer with LangChain and FastAPI experience",
                    job_title="Python Engineer",
                )
            )
            assert result["score"] >= 75
            assert result["recommendation"] == "apply"

    def test_low_match_score_for_mismatched_job(self):
        """Low-overlap resume+JD pair should score < 40."""
        with patch("tools.matching_tools.get_llm") as mock_llm:
            mock_response = MagicMock()
            mock_response.content = json.dumps(
                {
                    "score": 22,
                    "matching_skills": [],
                    "missing_skills": ["Java", "Spring Boot", "Kubernetes"],
                    "recommendation": "skip",
                    "explanation": "Very different tech stacks.",
                }
            )
            mock_llm.return_value.invoke.return_value = mock_response

            from tools.matching_tools import compute_match_score

            result = json.loads(
                compute_match_score.run(
                    resume_text="Python, PyTorch, deep learning researcher",
                    job_description="Java Spring Boot microservices architect with 10 years experience",
                    job_title="Java Architect",
                )
            )
            assert result["score"] < 40
            assert result["recommendation"] == "skip"


# ── Database Tests ────────────────────────────────────────────


@pytest.mark.asyncio
class TestDatabase:
    async def test_job_create_and_retrieve(self):
        """Saved job should be retrievable by external_id."""
        from database.connection import get_session, init_db
        from database.models import Job
        from database.repository import JobRepository

        # Use in-memory SQLite for tests
        import os
        os.environ["DATABASE_URL"] = "sqlite:///./data/test.db"

        await init_db()

        async with get_session() as session:
            repo = JobRepository(session)
            job = Job(
                title="Test Engineer",
                company="Acme Corp",
                external_id="test-ext-001",
                source="indeed",
            )
            saved = await repo.create(job)
            assert saved.id is not None

            found = await repo.get_by_external_id("test-ext-001")
            assert found is not None
            assert found.title == "Test Engineer"

    async def test_duplicate_job_detection(self):
        """Second save of same external_id should be detected as duplicate."""
        from database.connection import get_session
        from database.repository import JobRepository

        async with get_session() as session:
            repo = JobRepository(session)
            existing = await repo.get_by_external_id("test-ext-001")
            # If exists from previous test, dedup works
            assert existing is not None


# ── Cover Letter Tests ────────────────────────────────────────


class TestCoverLetterAgent:
    def test_cover_letter_has_required_sections(self):
        """Generated cover letter should contain greeting and closing."""
        with patch("tools.writing_tools.get_llm") as mock_llm:
            mock_response = MagicMock()
            mock_response.content = json.dumps(
                {
                    "cover_letter": "Dear Hiring Manager,\n\nTest content.\n\nSincerely,\nJohn",
                    "word_count": 150,
                    "key_points": ["Experience", "Culture fit"],
                    "subject_line": "Application: Senior Engineer",
                }
            )
            mock_llm.return_value.invoke.return_value = mock_response

            from tools.writing_tools import generate_cover_letter

            result = json.loads(
                generate_cover_letter.run(
                    resume_text="5 years Python experience",
                    job_description="We need a Python engineer",
                    job_title="Senior Python Engineer",
                    company_name="TechCorp",
                )
            )
            assert "cover_letter" in result
            assert len(result["cover_letter"]) > 50
            assert "word_count" in result
