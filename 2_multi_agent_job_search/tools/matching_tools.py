"""
tools/matching_tools.py
────────────────────────
Tools for semantic job-resume matching using vector similarity + LLM analysis.
Used by the Job Matching Agent.
"""

from __future__ import annotations

import json
from typing import Any

from crewai.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from utils.llm_factory import get_llm
from vectorstore.store import vector_store


@tool("compute_match_score")
def compute_match_score(
    resume_text: str,
    job_description: str,
    job_title: str = "",
    job_id: str = "",
) -> str:
    """
    Compute a semantic match score (0–100) between a resume and job description.
    Returns detailed analysis including matched skills, missing skills, and recommendation.

    Args:
        resume_text:     Full text of the candidate's resume.
        job_description: Full text of the job posting.
        job_title:       Job title for context (optional).
        job_id:          Job database ID for reference (optional).

    Returns:
        JSON with score, explanation, matching_skills, missing_skills, recommendation.
    """
    llm = get_llm(temperature=0.0)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert technical recruiter with 15 years of experience in AI/ML and software engineering hiring.

Analyse the match between the candidate's resume and the job description. 
Be rigorous and objective.

Return ONLY a valid JSON object:
{{
  "score": <integer 0-100>,
  "score_breakdown": {{
    "technical_skills": <0-40>,
    "experience_level": <0-25>,
    "domain_relevance": <0-20>,
    "education": <0-10>,
    "soft_skills": <0-5>
  }},
  "matching_skills": ["<skill1>", ...],
  "missing_skills": ["<skill1>", ...],
  "experience_match": "<strong|moderate|weak>",
  "explanation": "<2-3 sentence summary>",
  "recommendation": "<apply|consider|skip>",
  "cover_letter_angle": "<suggested narrative angle for cover letter>",
  "interview_prep_focus": ["<topic1>", ...]
}}

Score guide:
- 80-100: Excellent match — definitely apply
- 60-79:  Good match — worth applying
- 40-59:  Moderate match — apply with strong cover letter
- 20-39:  Weak match — focus on skill gaps first
- 0-19:   Poor match — not recommended
""",
            ),
            (
                "human",
                """Job Title: {job_title}

Job Description:
{job_description}

Candidate Resume:
{resume_text}

Provide match analysis:""",
            ),
        ]
    )

    chain = prompt | llm
    try:
        response = chain.invoke(
            {
                "job_title": job_title,
                "job_description": job_description[:4000],
                "resume_text": resume_text[:4000],
            }
        )
        content = response.content.strip().lstrip("```json").rstrip("```").strip()
        result = json.loads(content)
        result["job_id"] = job_id
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Match scoring failed: {e}")
        return json.dumps({"score": 0, "error": str(e), "job_id": job_id})


@tool("rank_jobs_by_match")
def rank_jobs_by_match(resume_text: str, jobs_json: str, top_n: int = 10) -> str:
    """
    Rank a list of jobs by their semantic similarity to the resume.
    Uses vector store for fast ranking, then LLM for top-N detailed scoring.

    Args:
        resume_text: Candidate's resume text.
        jobs_json:   JSON array of job dicts (must include 'description' and 'id' fields).
        top_n:       How many top matches to return with detailed scores.

    Returns:
        JSON with sorted list of {job_id, score, explanation}.
    """
    try:
        jobs: list[dict[str, Any]] = json.loads(jobs_json)
    except Exception as e:
        return json.dumps({"error": f"Invalid jobs_json: {e}"})

    if not jobs:
        return json.dumps({"ranked": [], "total": 0})

    # Fast vector similarity pass (no LLM cost)
    scored_fast: list[dict[str, Any]] = []
    for job in jobs:
        desc = job.get("description", "") or job.get("title", "")
        if not desc:
            continue
        try:
            docs_scores = vector_store.similarity_score(resume_text[:1000], collection="jobs")
            # Fallback: use job description directly for scoring
            score_rough = max(0, 100 - (min(docs_scores[0][1], 1.0) * 100)) if docs_scores else 50
        except Exception:
            score_rough = 50

        scored_fast.append({"job_id": job.get("id", ""), "job": job, "rough_score": score_rough})

    # Sort and take top_n for expensive LLM pass
    scored_fast.sort(key=lambda x: x["rough_score"], reverse=True)
    top_candidates = scored_fast[:top_n]

    final_ranked: list[dict[str, Any]] = []
    for item in top_candidates:
        job = item["job"]
        match_result_str = compute_match_score(
            resume_text=resume_text,
            job_description=job.get("description", job.get("title", "")),
            job_title=job.get("title", ""),
            job_id=job.get("id", ""),
        )
        match_result = json.loads(match_result_str)
        final_ranked.append(
            {
                "job_id": job.get("id", ""),
                "title": job.get("title", ""),
                "company": job.get("company", ""),
                "score": match_result.get("score", 0),
                "recommendation": match_result.get("recommendation", ""),
                "explanation": match_result.get("explanation", ""),
                "missing_skills": match_result.get("missing_skills", []),
                "matching_skills": match_result.get("matching_skills", []),
            }
        )

    final_ranked.sort(key=lambda x: x["score"], reverse=True)

    return json.dumps(
        {
            "ranked": final_ranked,
            "total_jobs_evaluated": len(jobs),
            "top_n_detailed": len(final_ranked),
        }
    )


@tool("identify_skill_gaps")
def identify_skill_gaps(resume_text: str, target_role: str) -> str:
    """
    Analyse skill gaps between the candidate's resume and a target role.
    Provides a personalised learning roadmap.

    Args:
        resume_text:  Candidate's resume text.
        target_role:  Target job title/role (e.g. "Senior ML Engineer at FAANG").

    Returns:
        JSON with current_skills, missing_skills, learning_roadmap, timeline_estimate.
    """
    llm = get_llm(temperature=0.1)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a senior engineering career coach.
Analyse the candidate's skills against requirements for the target role.

Return JSON:
{{
  "target_role": "<role>",
  "current_skills": ["<skill>"],
  "strong_areas": ["<area>"],
  "missing_critical": ["<must-have skills>"],
  "missing_nice_to_have": ["<good-to-have skills>"],
  "learning_roadmap": [
    {{"skill": "<name>", "resources": ["<url or book>"], "estimated_weeks": <n>}}
  ],
  "total_readiness_percent": <0-100>,
  "estimated_months_to_ready": <n>,
  "immediate_actions": ["<actionable step>"]
}}""",
            ),
            ("human", "Target Role: {role}\n\nResume:\n{resume}"),
        ]
    )
    chain = prompt | llm
    try:
        response = chain.invoke({"role": target_role, "resume": resume_text[:5000]})
        content = response.content.strip().lstrip("```json").rstrip("```")
        return content
    except Exception as e:
        logger.error(f"Skill gap analysis failed: {e}")
        return json.dumps({"error": str(e)})
