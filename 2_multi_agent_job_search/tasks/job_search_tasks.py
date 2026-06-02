"""
tasks/job_search_tasks.py
──────────────────────────
Task definitions for all CrewAI workflow stages.
Tasks are composed into Crews in crews/workflows.py.
"""

from __future__ import annotations

from crewai import Agent, Task


def research_jobs_task(agent: Agent, keywords: str, location: str) -> Task:
    return Task(
        description=f"""
Search for job openings matching the following criteria:
- Keywords: {keywords}
- Location: {location}
- Sources: LinkedIn, Indeed, Wellfound (search all three)
- Max jobs per source: 30

For each job, extract:
1. Job title (exact)
2. Company name
3. Location / remote status
4. Salary range (if listed)
5. Required skills (as a list)
6. Experience level (entry/mid/senior)
7. Application URL
8. Source platform

After collecting all results:
- Remove duplicate jobs (same title + company)
- Sort by posting recency
- Return consolidated JSON array

Be thorough. Do not stop at one source.
""",
        expected_output=(
            "A JSON object with key 'jobs' containing an array of deduplicated job listings. "
            "Each job has: title, company, location, is_remote, salary, required_skills, "
            "experience_level, application_url, source, external_id."
        ),
        agent=agent,
    )


def analyse_resume_task(agent: Agent, file_path: str) -> Task:
    return Task(
        description=f"""
Parse and analyse the resume at: {file_path}

Steps:
1. Extract all text from the file (handle PDF and DOCX)
2. Structure the data: name, contact info, skills, experience, education, projects
3. Calculate total years of experience
4. Identify top 10 technical skills
5. Create a concise professional summary
6. Generate and store embeddings for semantic matching
7. Return the complete structured profile

Be precise with dates and skill extraction.
""",
        expected_output=(
            "A JSON object with: raw_text, parsed (structured fields), char_count. "
            "Parsed includes: name, email, skills (list), experience (list of roles with dates), "
            "education, projects, total_years_experience, summary."
        ),
        agent=agent,
    )


def match_jobs_task(agent: Agent, context_tasks: list[Task]) -> Task:
    return Task(
        description="""
Using the resume data and job listings from previous tasks:

1. Compute a match score (0-100) for each job against the resume
2. For each job provide:
   - Numeric match score
   - Score breakdown (technical skills, experience, domain, education)
   - List of matching skills
   - List of missing/gap skills
   - Recommendation: apply / consider / skip
   - 1-sentence explanation

3. Rank all jobs by match score (highest first)
4. Flag any jobs with score >= 75 as "hot matches"
5. Identify the top 3 skill gaps across all jobs

Return the full ranked list with analysis.
""",
        expected_output=(
            "A JSON object with: ranked_jobs (array sorted by score desc), "
            "hot_matches (jobs with score >= 75), top_skill_gaps (list), "
            "total_jobs_evaluated (count)."
        ),
        agent=agent,
        context=context_tasks,
    )


def customize_resume_task(
    agent: Agent, job_title: str, company: str, context_tasks: list[Task]
) -> Task:
    return Task(
        description=f"""
Customise the candidate's resume specifically for:
Job: {job_title}
Company: {company}

Using the resume text and match analysis from previous tasks:

1. Identify keywords from the job description not present in the resume
2. Rewrite bullet points to emphasise relevant achievements
3. Reorder sections: put most relevant experience first
4. Add ATS-friendly keywords naturally (not keyword stuffing)
5. Address top 2-3 skill gaps tastefully (if learnable, mention "currently upskilling in X")
6. Maintain truthfulness — do NOT add fake experience

Output: complete tailored resume in Markdown format + list of changes made.
""",
        expected_output=(
            "A JSON object with: customized_resume (Markdown string), "
            "changes_made (list of specific changes), ats_keywords_added (list)."
        ),
        agent=agent,
        context=context_tasks,
    )


def write_cover_letter_task(
    agent: Agent, job_title: str, company: str, context_tasks: list[Task]
) -> Task:
    return Task(
        description=f"""
Write a compelling cover letter for:
Job: {job_title}
Company: {company}

Steps:
1. Research {company}: mission, products, culture, recent news
2. Identify the strongest 2-3 achievements from the resume that align with this role
3. Write the cover letter:
   - Paragraph 1: Hook + why this specific company
   - Paragraph 2: Most relevant achievement + direct impact
   - Paragraph 3: Why you're uniquely suited (skills + culture fit)
   - Paragraph 4: Confident call to action
4. Length: 300-380 words
5. Tone: Professional but warm and direct

Also prepare 5 likely interview questions with answer frameworks.
""",
        expected_output=(
            "A JSON object with: cover_letter (string), word_count (int), "
            "key_points (list), subject_line (string), "
            "interview_prep (object with likely_questions list)."
        ),
        agent=agent,
        context=context_tasks,
    )


def track_applications_task(
    agent: Agent, jobs_json: str, context_tasks: list[Task]
) -> Task:
    return Task(
        description=f"""
Save all discovered jobs and application data to the database.

Jobs JSON: {jobs_json[:200]}...

Steps:
1. Save all discovered jobs (deduplicate by external_id)
2. For any jobs where cover letters were generated, create APPLICATION records
3. Set initial status to APPLIED for submitted applications
4. Retrieve and return application statistics:
   - Total jobs discovered
   - Applications by status
   - Response rate
   - Top sources
5. Export a CSV report to ./data/reports/applications.csv

Return the statistics summary.
""",
        expected_output=(
            "A JSON object with: applications_by_status (dict), "
            "jobs_by_source (dict), total_jobs_discovered, total_applications, "
            "response_rate, csv_export_path."
        ),
        agent=agent,
        context=context_tasks,
    )


def send_notifications_task(
    agent: Agent, context_tasks: list[Task]
) -> Task:
    return Task(
        description="""
Based on the job search results and match scores from previous tasks:

1. Identify all jobs with match score >= 70 ("high match" threshold)
2. Send a new-jobs alert email with the top matches
3. Generate a weekly analytics report including:
   - Jobs discovered this week
   - Applications by status  
   - Top matching companies
   - Skill gap summary
   - Recommended actions for next week
4. Send the weekly report via email

Return confirmation of all notifications sent.
""",
        expected_output=(
            "A JSON object with: email_sent (bool), jobs_in_alert (count), "
            "report_generated (bool), report_preview (first 200 chars of report)."
        ),
        agent=agent,
        context=context_tasks,
    )
