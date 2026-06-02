"""
tools/resume_tools.py
──────────────────────
Tools for reading, parsing, and embedding resumes (PDF / DOCX).
Used by the Resume Analysis Agent and Resume Customization Agent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from crewai.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from config import settings
from utils.llm_factory import get_llm
from vectorstore.store import vector_store


# ── Parsing helpers ───────────────────────────────────────────


def _extract_pdf_text(path: str) -> str:
    """Extract plain text from a PDF using pdfplumber (best for text-based PDFs)."""
    import pdfplumber

    text_parts: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return "\n\n".join(text_parts)


def _extract_docx_text(path: str) -> str:
    """Extract plain text from a .docx file."""
    from docx import Document

    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


# ── Tools ────────────────────────────────────────────────────


@tool("parse_resume")
def parse_resume(file_path: str) -> str:
    """
    Parse a resume PDF or DOCX file.
    Returns a JSON string with:
    - raw_text
    - structured fields: name, email, phone, skills, experience, education, projects

    Args:
        file_path: Absolute or relative path to the resume file.
    """
    path = Path(file_path)
    if not path.exists():
        return json.dumps({"error": f"File not found: {file_path}"})

    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            raw_text = _extract_pdf_text(str(path))
        elif ext in (".docx", ".doc"):
            raw_text = _extract_docx_text(str(path))
        else:
            return json.dumps({"error": f"Unsupported file type: {ext}"})
    except Exception as e:
        logger.error(f"Text extraction failed for {file_path}: {e}")
        return json.dumps({"error": str(e)})

    # Use LLM to structure the extracted text
    llm = get_llm(temperature=0.0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a precise resume parser. Extract structured information from the resume text.
Return ONLY a valid JSON object with these keys:
- name (string)
- email (string)
- phone (string)
- location (string)
- linkedin_url (string or null)
- github_url (string or null)
- summary (string — 2-3 sentence professional summary)
- skills (list of strings — technical skills only)
- soft_skills (list of strings)
- experience (list of objects: {company, title, start_date, end_date, description, achievements})
- education (list of objects: {institution, degree, field, graduation_year, gpa})
- projects (list of objects: {name, description, technologies, url})
- certifications (list of strings)
- languages (list of strings)
- total_years_experience (number)
""",
            ),
            ("human", "Parse this resume:\n\n{resume_text}"),
        ]
    )
    chain = prompt | llm
    try:
        response = chain.invoke({"resume_text": raw_text[:8000]})
        structured = json.loads(response.content)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"LLM parsing failed, returning raw text only: {e}")
        structured = {}

    result = {
        "file_path": str(path),
        "raw_text": raw_text,
        "parsed": structured,
        "char_count": len(raw_text),
    }
    logger.info(f"Resume parsed: {path.name} ({len(raw_text)} chars)")
    return json.dumps(result)


@tool("embed_resume")
def embed_resume(resume_id: str, raw_text: str, metadata: str = "{}") -> str:
    """
    Create vector embeddings from resume text and store in the vector store.

    Args:
        resume_id: Unique identifier for this resume (UUID).
        raw_text:  Full plain-text content of the resume.
        metadata:  JSON string of extra metadata to store alongside the embedding.

    Returns:
        JSON with status and chunk count.
    """
    meta: dict[str, Any] = {}
    try:
        meta = json.loads(metadata)
    except Exception:
        pass

    try:
        vector_store.add_resume(resume_id, raw_text, meta)
        return json.dumps({"status": "ok", "resume_id": resume_id, "text_length": len(raw_text)})
    except Exception as e:
        logger.error(f"Embedding resume {resume_id} failed: {e}")
        return json.dumps({"status": "error", "error": str(e)})


@tool("customize_resume")
def customize_resume(
    original_resume_text: str,
    job_description: str,
    missing_skills: str = "[]",
) -> str:
    """
    Rewrite and tailor a resume for a specific job description.
    Adds ATS-friendly keywords, adjusts bullet points, highlights relevant experience.

    Args:
        original_resume_text: Full text of the candidate's resume.
        job_description:      Full text of the target job description.
        missing_skills:       JSON list of skills the candidate is missing.

    Returns:
        JSON with 'customized_resume' (markdown-formatted) and 'changes_made' (list).
    """
    llm = get_llm(temperature=0.2)
    skills_list = json.loads(missing_skills) if missing_skills else []

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert resume writer specialising in ATS optimisation.
Your task: rewrite the candidate's resume to maximise relevance for the given job description.

Rules:
1. Preserve all factual information — never fabricate experience or skills.
2. Incorporate exact keywords from the job description naturally.
3. Reorder sections to lead with most relevant experience.
4. Strengthen action verbs and quantify achievements where possible.
5. If the candidate lacks some required skills, acknowledge them tastefully (e.g. "Currently learning X").
6. Output format: Markdown with clear section headers.

Return a JSON object:
{{
  "customized_resume": "<full markdown resume>",
  "changes_made": ["<list of specific changes>"],
  "ats_keywords_added": ["<keywords inserted>"]
}}
""",
            ),
            (
                "human",
                """Original Resume:
{resume}

Job Description:
{jd}

Missing Skills to Address: {missing}""",
            ),
        ]
    )
    chain = prompt | llm
    try:
        response = chain.invoke(
            {
                "resume": original_resume_text[:6000],
                "jd": job_description[:3000],
                "missing": ", ".join(skills_list),
            }
        )
        # Strip potential markdown fences
        content = response.content.strip().lstrip("```json").rstrip("```")
        return content
    except Exception as e:
        logger.error(f"Resume customization failed: {e}")
        return json.dumps({"error": str(e), "customized_resume": original_resume_text})
