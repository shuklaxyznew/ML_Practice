"""
tools/notification_tools.py
─────────────────────────────
Email notification and alerting tools using SendGrid.
"""

from __future__ import annotations

import json
from datetime import datetime

from crewai.tools import tool
from loguru import logger

from config import settings


@tool("send_email_notification")
def send_email_notification(
    subject: str,
    body: str,
    recipient: str = "",
    is_html: bool = False,
) -> str:
    """
    Send an email notification via SendGrid.

    Args:
        subject:   Email subject line.
        body:      Email body (plain text or HTML).
        recipient: Override recipient email (uses env default if empty).
        is_html:   True if body is HTML, False for plain text.

    Returns:
        JSON with status and message_id.
    """
    if not settings.enable_email_notifications:
        logger.info(f"Email notifications disabled. Would have sent: {subject}")
        return json.dumps({"status": "disabled", "subject": subject})

    to_email = recipient or settings.notification_email_to
    if not to_email:
        return json.dumps({"status": "error", "error": "No recipient configured"})

    try:
        import sendgrid
        from sendgrid.helpers.mail import Content, Email, Mail, To

        sg = sendgrid.SendGridAPIClient(api_key=settings.sendgrid_api_key)
        mail = Mail(
            from_email=Email(settings.notification_email_from),
            to_emails=To(to_email),
            subject=subject,
            html_content=Content("text/html", body) if is_html else None,
            plain_text_content=Content("text/plain", body) if not is_html else None,
        )
        response = sg.client.mail.send.post(request_body=mail.get())
        msg_id = response.headers.get("X-Message-Id", "unknown")
        logger.info(f"Email sent: {subject} → {to_email} (id={msg_id})")
        return json.dumps({"status": "sent", "message_id": msg_id, "to": to_email})

    except Exception as e:
        logger.error(f"Email send failed: {e}")
        return json.dumps({"status": "error", "error": str(e)})


@tool("send_new_jobs_alert")
def send_new_jobs_alert(jobs_json: str, min_score: float = 70.0) -> str:
    """
    Send an alert email listing newly discovered high-match jobs.

    Args:
        jobs_json:  JSON array of job dicts (with 'score', 'title', 'company', 'application_url').
        min_score:  Only include jobs with match score >= this threshold.

    Returns:
        JSON with status and count of jobs included.
    """
    try:
        jobs = json.loads(jobs_json)
    except Exception:
        return json.dumps({"status": "error", "error": "Invalid jobs_json"})

    high_match = [j for j in jobs if j.get("score", 0) >= min_score]
    if not high_match:
        return json.dumps({"status": "skipped", "reason": "No jobs above score threshold"})

    rows = "\n".join(
        f"  • {j['title']} at {j['company']} — Score: {j.get('score', 'N/A')}/100\n"
        f"    Apply: {j.get('application_url', 'N/A')}"
        for j in high_match[:10]
    )

    subject = f"🎯 {len(high_match)} New High-Match Jobs Found!"
    body = f"""Job Search Update — {datetime.now().strftime('%B %d, %Y')}

You have {len(high_match)} new jobs that match your profile at 70%+:

{rows}

Log in to your dashboard to view full details, cover letters, and customised resumes.

Good luck! 🚀
"""
    return send_email_notification(subject=subject, body=body)


@tool("track_application_status_change")
def track_application_status_change(
    company: str,
    job_title: str,
    old_status: str,
    new_status: str,
) -> str:
    """
    Send a notification when an application status changes (e.g. APPLIED → INTERVIEW).

    Args:
        company:    Company name.
        job_title:  Job title.
        old_status: Previous application status.
        new_status: Updated application status.

    Returns:
        JSON notification result.
    """
    emoji_map = {
        "applied": "📤",
        "phone_screen": "📞",
        "interview": "🤝",
        "final_round": "🏆",
        "offer": "🎉",
        "rejected": "❌",
        "withdrawn": "↩️",
    }
    emoji = emoji_map.get(new_status.lower(), "📋")

    subject = f"{emoji} Application Update: {job_title} at {company}"
    body = f"""Application Status Update

Job:    {job_title} at {company}
Status: {old_status.upper()} → {new_status.upper()}
Date:   {datetime.now().strftime('%Y-%m-%d %H:%M')}

{"Congratulations! 🎊 This is great news." if new_status in ('interview', 'offer', 'final_round') else ""}
{"Don't forget to prepare for your interview!" if new_status == 'interview' else ""}
{"Review your offer carefully and consider negotiating." if new_status == 'offer' else ""}
"""
    return send_email_notification(subject=subject, body=body)
