"""
tools/writing_tools.py
───────────────────────
Tools for generating cover letters, interview prep materials, and reports.
"""

from __future__ import annotations

import json

from crewai.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from utils.llm_factory import get_llm


@tool("generate_cover_letter")
def generate_cover_letter(
    resume_text: str,
    job_description: str,
    job_title: str,
    company_name: str,
    company_context: str = "",
    tone: str = "professional",
    hiring_manager: str = "Hiring Manager",
) -> str:
    """
    Generate a personalised, ATS-optimised cover letter.

    Args:
        resume_text:     Candidate's resume text.
        job_description: Full job description text.
        job_title:       Exact job title.
        company_name:    Company name.
        company_context: Brief company info (mission, products, culture).
        tone:            "professional", "enthusiastic", or "conversational".
        hiring_manager:  Name of hiring manager if known.

    Returns:
        JSON with 'cover_letter' (markdown), 'word_count', and 'key_points'.
    """
    llm = get_llm(temperature=0.4)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""You are an expert cover letter writer known for crafting compelling, 
human-sounding letters that get candidates to interviews.

Tone: {tone}

Rules:
1. Open with a strong hook — NOT "I am writing to apply..."
2. Connect candidate's specific achievements to the company's specific needs.
3. Show genuine enthusiasm for the company/mission.
4. Keep to 3-4 paragraphs, ~300-400 words.
5. Close with a confident, specific call to action.
6. Never sound like AI wrote it — be warm, specific, and direct.

Return JSON:
{{
  "cover_letter": "<full cover letter text>",
  "word_count": <integer>,
  "key_points": ["<point covered>", ...],
  "subject_line": "<email subject if sending directly>"
}}""",
            ),
            (
                "human",
                """Write a cover letter for:

Job: {job_title} at {company_name}
Hiring Manager: {hiring_manager}

Company Context:
{company_context}

Job Description:
{job_description}

Candidate Resume:
{resume_text}""",
            ),
        ]
    )
    chain = prompt | llm
    try:
        response = chain.invoke(
            {
                "job_title": job_title,
                "company_name": company_name,
                "hiring_manager": hiring_manager,
                "company_context": company_context[:1000],
                "job_description": job_description[:3000],
                "resume_text": resume_text[:3000],
            }
        )
        content = response.content.strip().lstrip("```json").rstrip("```").strip()
        return content
    except Exception as e:
        logger.error(f"Cover letter generation failed: {e}")
        return json.dumps({"error": str(e), "cover_letter": ""})


@tool("generate_interview_prep")
def generate_interview_prep(
    resume_text: str,
    job_description: str,
    job_title: str,
    company_name: str,
    interview_round: str = "general",
) -> str:
    """
    Generate personalised interview preparation materials.

    Args:
        resume_text:      Candidate's resume.
        job_description:  Job posting text.
        job_title:        Role title.
        company_name:     Company being interviewed at.
        interview_round:  "phone_screen", "technical", "system_design", "behavioural", or "general".

    Returns:
        JSON with likely_questions (with suggested answers), topics_to_review, talking_points.
    """
    llm = get_llm(temperature=0.3)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""You are an expert interview coach for top tech companies.
Prepare the candidate specifically for a {interview_round} interview.

Return JSON:
{{
  "interview_round": "{interview_round}",
  "likely_questions": [
    {{
      "question": "<question>",
      "category": "<technical|behavioural|situational|company>",
      "suggested_answer_framework": "<STAR/SOAR/specific approach>",
      "key_points_to_cover": ["<point>"]
    }}
  ],
  "topics_to_review": ["<topic>"],
  "resume_talking_points": ["<strong experience to highlight>"],
  "questions_to_ask_interviewer": ["<thoughtful question>"],
  "red_flags_to_avoid": ["<common mistake>"],
  "company_research_points": ["<thing to know about {company_name}>"]
}}""",
            ),
            (
                "human",
                "Job: {job_title} at {company_name}\n\nJob Description:\n{jd}\n\nResume:\n{resume}",
            ),
        ]
    )
    chain = prompt | llm
    try:
        response = chain.invoke(
            {
                "job_title": job_title,
                "company_name": company_name,
                "jd": job_description[:3000],
                "resume": resume_text[:3000],
            }
        )
        content = response.content.strip().lstrip("```json").rstrip("```").strip()
        return content
    except Exception as e:
        logger.error(f"Interview prep generation failed: {e}")
        return json.dumps({"error": str(e)})


@tool("research_company")
def research_company(company_name: str, job_title: str = "") -> str:
    """
    Generate a company research brief to inform cover letters and interview prep.
    Uses LLM knowledge — for real-time data, pair with a web search tool.

    Args:
        company_name: Company to research.
        job_title:    Role being applied to (for relevance filtering).

    Returns:
        JSON with company profile, culture, recent news, and strategic context.
    """
    llm = get_llm(temperature=0.2)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a business analyst preparing a company brief for a job applicant.

Return JSON:
{
  "company_name": "<name>",
  "industry": "<industry>",
  "founded": "<year>",
  "size_estimate": "<employees>",
  "products_services": ["<product/service>"],
  "mission_statement": "<mission>",
  "tech_stack_known": ["<technologies>"],
  "culture_keywords": ["<keyword>"],
  "recent_developments": ["<event/news>"],
  "competitors": ["<competitor>"],
  "why_join": ["<compelling reason to work there>"],
  "potential_challenges": ["<honest consideration>"],
  "ipo_status": "<public|private|acquired>",
  "notable_investors": ["<investor>"]
}""",
            ),
            (
                "human",
                "Research {company} for a {role} role. Provide what you know up to your knowledge cutoff.",
            ),
        ]
    )
    chain = prompt | llm
    try:
        response = chain.invoke({"company": company_name, "role": job_title or "engineering"})
        content = response.content.strip().lstrip("```json").rstrip("```").strip()
        return content
    except Exception as e:
        logger.error(f"Company research failed for {company_name}: {e}")
        return json.dumps({"error": str(e), "company_name": company_name})


@tool("generate_weekly_report")
def generate_weekly_report(stats_json: str) -> str:
    """
    Generate a formatted weekly job search analytics report.

    Args:
        stats_json: JSON string with application statistics.

    Returns:
        Markdown-formatted report string.
    """
    llm = get_llm(temperature=0.2)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a career advisor generating a weekly job search report.
Write a concise, encouraging, data-driven report in Markdown.
Include:
- Executive summary
- Key metrics table
- Week-over-week trends
- Top matching jobs
- Recommended actions for next week
- Motivational closing note""",
            ),
            ("human", "Generate a weekly report from these stats:\n\n{stats}"),
        ]
    )
    chain = prompt | llm
    try:
        response = chain.invoke({"stats": stats_json})
        return json.dumps({"report": response.content, "format": "markdown"})
    except Exception as e:
        logger.error(f"Weekly report generation failed: {e}")
        return json.dumps({"error": str(e), "report": ""})
