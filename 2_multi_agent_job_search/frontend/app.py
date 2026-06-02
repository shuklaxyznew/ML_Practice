"""
frontend/app.py
────────────────
Streamlit dashboard for the Multi-Agent Job Search Assistant.

Run:  streamlit run frontend/app.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from database.connection import get_session, init_db
from database.models import Application, ApplicationStatus, Job
from database.repository import ApplicationRepository, JobRepository, MatchScoreRepository

# ── Page Config ──────────────────────────────────────────────

st.set_page_config(
    page_title="AI Job Search Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/yourusername/multi-agent-job-search",
        "About": "Multi-Agent AI Job Search Assistant — Built with CrewAI & LangChain",
    },
)

# ── Custom CSS ───────────────────────────────────────────────

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --primary: #6366f1;
    --primary-dark: #4f46e5;
    --secondary: #10b981;
    --danger: #ef4444;
    --warning: #f59e0b;
    --surface: #1e1e2e;
    --surface-2: #2a2a3e;
    --border: #3f3f5a;
    --text: #e2e8f0;
    --text-muted: #94a3b8;
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
}

.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #0f0f1a 100%);
    color: var(--text);
}

/* Sidebar */
.css-1d391kg, [data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* Cards */
.metric-card {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: var(--primary); }
.metric-value { font-size: 2.5rem; font-weight: 700; color: var(--primary); }
.metric-label { font-size: 0.85rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }

/* Score badge */
.score-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.score-high { background: rgba(16,185,129,0.2); color: #10b981; border: 1px solid #10b981; }
.score-mid  { background: rgba(245,158,11,0.2); color: #f59e0b; border: 1px solid #f59e0b; }
.score-low  { background: rgba(239,68,68,0.2);  color: #ef4444; border: 1px solid #ef4444; }

/* Status pill */
.status-pill {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 500;
}

/* Job card */
.job-card {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s, transform 0.1s;
}
.job-card:hover { border-color: var(--primary); transform: translateY(-1px); }

/* Agent status */
.agent-running {
    animation: pulse 1.5s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

h1, h2, h3 { color: var(--text) !important; }
.stButton button {
    background: var(--primary) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
}
.stButton button:hover { background: var(--primary-dark) !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Helpers ──────────────────────────────────────────────────


def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def score_class(score: float) -> str:
    if score >= 75:
        return "score-high"
    if score >= 50:
        return "score-mid"
    return "score-low"


STATUS_COLORS = {
    "discovered": "#94a3b8",
    "applied": "#6366f1",
    "phone_screen": "#8b5cf6",
    "interview": "#10b981",
    "final_round": "#f59e0b",
    "offer": "#22c55e",
    "rejected": "#ef4444",
    "withdrawn": "#64748b",
}

STATUS_EMOJI = {
    "discovered": "🔍",
    "applied": "📤",
    "phone_screen": "📞",
    "interview": "🤝",
    "final_round": "🏆",
    "offer": "🎉",
    "rejected": "❌",
    "withdrawn": "↩️",
}


# ── Sidebar ──────────────────────────────────────────────────


def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div style='text-align:center; padding: 1rem 0;'>
                <div style='font-size:2.5rem'>🤖</div>
                <h2 style='margin:0; color:#6366f1;'>Job Search AI</h2>
                <p style='font-size:0.8rem; color:#94a3b8; margin:0;'>Multi-Agent Assistant</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown("**Navigation**")
        page = st.radio(
            "",
            [
                "🏠 Dashboard",
                "🔍 Job Search",
                "📄 Resume",
                "🎯 Matches",
                "📊 Applications",
                "📬 Cover Letters",
                "📈 Analytics",
                "⚙️ Settings",
            ],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("**Agent Status**")

        agents = [
            ("🔎 Job Research", "idle"),
            ("📋 Resume Analysis", "idle"),
            ("🧠 Job Matching", "idle"),
            ("✏️ Customization", "idle"),
            ("📝 Cover Letter", "idle"),
            ("📁 App Tracking", "idle"),
            ("🔔 Notifications", "idle"),
        ]
        for name, status in agents:
            color = "#10b981" if status == "running" else "#64748b"
            st.markdown(
                f"<small style='color:{color};'>● {name}</small>",
                unsafe_allow_html=True,
            )

    return page


# ── Pages ────────────────────────────────────────────────────


def page_dashboard():
    st.title("🤖 AI Job Search Assistant")
    st.markdown(
        "<p style='color:#94a3b8; font-size:1.1rem;'>Your autonomous multi-agent job search pipeline</p>",
        unsafe_allow_html=True,
    )

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("Total Jobs Found", "247", col1),
        ("Applications", "23", col2),
        ("Interviews", "5", col3),
        ("Avg Match Score", "72%", col4),
    ]
    for label, value, col in metrics:
        with col:
            st.markdown(
                f"""<div class='metric-card'>
                    <div class='metric-value'>{value}</div>
                    <div class='metric-label'>{label}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("📈 Application Pipeline")
        pipeline_data = {
            "Stage": ["Discovered", "Applied", "Phone Screen", "Interview", "Final Round", "Offer"],
            "Count": [247, 23, 8, 5, 2, 1],
            "Color": ["#94a3b8", "#6366f1", "#8b5cf6", "#10b981", "#f59e0b", "#22c55e"],
        }
        fig = go.Figure(
            go.Funnel(
                y=pipeline_data["Stage"],
                x=pipeline_data["Count"],
                textposition="inside",
                textinfo="value+percent initial",
                marker={"color": pipeline_data["Color"]},
            )
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e2e8f0", "family": "Space Grotesk"},
            margin=dict(l=0, r=0, t=10, b=0),
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("🎯 Top Matches Today")
        sample_matches = [
            {"title": "Senior ML Engineer", "company": "Anthropic", "score": 92},
            {"title": "AI Research Scientist", "company": "OpenAI", "score": 88},
            {"title": "Platform Engineer", "company": "Stripe", "score": 81},
            {"title": "Backend Engineer", "company": "Linear", "score": 75},
        ]
        for m in sample_matches:
            css = score_class(m["score"])
            st.markdown(
                f"""<div class='job-card'>
                    <strong>{m['title']}</strong><br>
                    <span style='color:#94a3b8; font-size:0.9rem;'>{m['company']}</span>
                    <span class='score-badge {css}' style='float:right;'>{m['score']}</span>
                </div>""",
                unsafe_allow_html=True,
            )

    # Recent activity
    st.subheader("⚡ Recent Activity")
    activities = [
        ("2m ago", "🔍", "Job Research Agent found 34 new listings on LinkedIn"),
        ("15m ago", "🧠", "Job Matching Agent scored 34 jobs — 8 high matches (≥75)"),
        ("1h ago", "📝", "Cover Letter generated for Senior ML Engineer @ Anthropic"),
        ("2h ago", "📤", "Application submitted: AI Research Scientist @ OpenAI"),
        ("1d ago", "🤝", "Interview scheduled: Platform Engineer @ Stripe"),
    ]
    for time_ago, emoji, action in activities:
        st.markdown(
            f"""<div style='display:flex; align-items:center; gap:1rem; 
                padding:0.6rem; border-radius:8px; margin-bottom:0.3rem;
                background:rgba(255,255,255,0.03); border:1px solid #2a2a3e;'>
                <span style='font-size:1.2rem'>{emoji}</span>
                <div>
                    <span style='color:#e2e8f0;'>{action}</span><br>
                    <span style='color:#64748b; font-size:0.8rem;'>{time_ago}</span>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )


def page_job_search():
    st.title("🔍 Job Search")
    st.markdown(
        "<p style='color:#94a3b8;'>Configure and launch your autonomous job search across multiple platforms</p>",
        unsafe_allow_html=True,
    )

    with st.form("search_form"):
        col1, col2 = st.columns(2)
        with col1:
            keywords = st.text_input(
                "Job Keywords",
                value="Python Engineer, ML Engineer, AI Engineer",
                help="Comma-separated keywords",
            )
            location = st.text_input("Location", value="Remote")
            experience = st.multiselect(
                "Experience Level",
                ["Intern", "Entry", "Mid", "Senior", "Staff", "Principal"],
                default=["Mid", "Senior"],
            )
        with col2:
            min_salary = st.number_input("Min Salary ($)", value=100000, step=10000)
            sources = st.multiselect(
                "Job Sources",
                ["LinkedIn", "Indeed", "Wellfound", "Company Sites"],
                default=["LinkedIn", "Indeed", "Wellfound"],
            )
            max_results = st.slider("Max Results per Source", 10, 100, 30)

        company_urls = st.text_area(
            "Company Career Pages (one URL per line — optional)",
            placeholder="https://jobs.ashbyhq.com/anthropic\nhttps://stripe.com/jobs",
            height=80,
        )

        submitted = st.form_submit_button("🚀 Launch AI Job Search", use_container_width=True)

    if submitted:
        progress = st.progress(0, text="Initialising agents...")
        status = st.empty()

        stages = [
            (10, "🔎 Job Research Agent: Searching LinkedIn..."),
            (25, "🔎 Job Research Agent: Searching Indeed..."),
            (40, "🔎 Job Research Agent: Searching Wellfound..."),
            (55, "🔎 Deduplicating and cleaning results..."),
            (70, "🧠 Job Matching Agent: Computing match scores..."),
            (85, "📁 Saving to database..."),
            (95, "🔔 Sending notifications..."),
            (100, "✅ Job search complete!"),
        ]

        for pct, msg in stages:
            time.sleep(0.4)
            progress.progress(pct, text=msg)
            status.markdown(f"<p style='color:#6366f1;'>{msg}</p>", unsafe_allow_html=True)

        st.success("✅ Job search complete! Found 34 new listings — 8 high matches.")
        st.balloons()


def page_resume():
    st.title("📄 Resume")
    st.markdown(
        "<p style='color:#94a3b8;'>Upload and analyse your resume</p>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Upload Resume")
        uploaded = st.file_uploader("Choose PDF or DOCX", type=["pdf", "docx"])

        if uploaded:
            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name

            with st.spinner("🤖 Parsing resume..."):
                time.sleep(1.5)  # Simulate parsing

            st.success(f"✅ Resume parsed: **{uploaded.name}**")
            st.markdown(
                """<div class='metric-card'>
                    <div class='metric-value'>8.5</div>
                    <div class='metric-label'>Years Experience</div>
                </div>""",
                unsafe_allow_html=True,
            )

    with col2:
        st.subheader("Extracted Profile")
        skills = ["Python", "PyTorch", "LangChain", "CrewAI", "FastAPI", "Docker", "AWS", "PostgreSQL"]
        st.markdown("**Technical Skills**")
        cols = st.columns(4)
        for i, skill in enumerate(skills):
            with cols[i % 4]:
                st.markdown(
                    f"<span style='background:#1e1e2e; border:1px solid #6366f1; border-radius:6px; "
                    f"padding:0.2rem 0.5rem; font-size:0.8rem; display:block; text-align:center; "
                    f"margin-bottom:0.3rem;'>{skill}</span>",
                    unsafe_allow_html=True,
                )

        st.markdown("**Experience**")
        st.markdown(
            """
- **Senior ML Engineer** — TechCorp (2022–Present)
- **Software Engineer** — StartupXYZ (2020–2022)
- **Junior Developer** — Agency (2018–2020)
"""
        )


def page_matches():
    st.title("🎯 Job Matches")
    st.markdown(
        "<p style='color:#94a3b8;'>AI-ranked job matches with detailed scoring</p>",
        unsafe_allow_html=True,
    )

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        min_score = st.slider("Min Match Score", 0, 100, 60)
    with col2:
        source_filter = st.multiselect("Source", ["linkedin", "indeed", "wellfound"], default=[])
    with col3:
        sort_by = st.selectbox("Sort By", ["Match Score", "Company", "Salary", "Date"])

    # Sample matches table
    matches = [
        {
            "Score": 92,
            "Title": "Senior ML Engineer",
            "Company": "Anthropic",
            "Location": "Remote",
            "Salary": "$180k–$240k",
            "Source": "linkedin",
            "Rec": "apply",
        },
        {
            "Score": 88,
            "Title": "AI Research Scientist",
            "Company": "OpenAI",
            "Location": "SF (Hybrid)",
            "Salary": "$200k–$280k",
            "Source": "wellfound",
            "Rec": "apply",
        },
        {
            "Score": 81,
            "Title": "Platform Engineer",
            "Company": "Stripe",
            "Location": "Remote",
            "Salary": "$160k–$220k",
            "Source": "linkedin",
            "Rec": "apply",
        },
        {
            "Score": 75,
            "Title": "Backend Engineer",
            "Company": "Linear",
            "Location": "Remote",
            "Salary": "$140k–$190k",
            "Source": "wellfound",
            "Rec": "apply",
        },
        {
            "Score": 68,
            "Title": "Data Engineer",
            "Company": "Airbnb",
            "Location": "SF",
            "Salary": "$150k–$200k",
            "Source": "indeed",
            "Rec": "consider",
        },
        {
            "Score": 55,
            "Title": "DevOps Engineer",
            "Company": "Shopify",
            "Location": "Remote",
            "Salary": "$130k–$170k",
            "Source": "indeed",
            "Rec": "consider",
        },
    ]

    filtered = [m for m in matches if m["Score"] >= min_score]
    if source_filter:
        filtered = [m for m in filtered if m["Source"] in source_filter]

    for m in filtered:
        css = score_class(m["Score"])
        rec_color = "#10b981" if m["Rec"] == "apply" else "#f59e0b"
        with st.container():
            st.markdown(
                f"""<div class='job-card'>
                    <div style='display:flex; justify-content:space-between; align-items:start;'>
                        <div>
                            <strong style='font-size:1.05rem;'>{m['Title']}</strong><br>
                            <span style='color:#94a3b8;'>{m['Company']} · {m['Location']}</span><br>
                            <span style='color:#64748b; font-size:0.85rem;'>{m['Salary']} · {m['Source']}</span>
                        </div>
                        <div style='text-align:right;'>
                            <span class='score-badge {css}'>{m['Score']}/100</span><br>
                            <span style='color:{rec_color}; font-size:0.8rem; font-weight:600;'>
                                {'✓ ' if m['Rec']=='apply' else '~ '}{m['Rec'].upper()}
                            </span>
                        </div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )


def page_applications():
    st.title("📊 Application Tracker")
    st.markdown(
        "<p style='color:#94a3b8;'>Track every application through the hiring pipeline</p>",
        unsafe_allow_html=True,
    )

    # Status columns (Kanban-like)
    statuses = ["Applied", "Phone Screen", "Interview", "Final Round", "Offer"]
    cols = st.columns(len(statuses))

    sample_apps = {
        "Applied": ["Senior ML Eng @ Anthropic", "AI Research Sci @ OpenAI", "Backend Eng @ Vercel"],
        "Phone Screen": ["Platform Eng @ Stripe", "Staff Eng @ Figma"],
        "Interview": ["ML Infra @ Meta", "AI Eng @ Cohere"],
        "Final Round": ["Senior Eng @ Linear"],
        "Offer": [],
    }

    for col, status in zip(cols, statuses):
        with col:
            color = STATUS_COLORS.get(status.lower().replace(" ", "_"), "#64748b")
            emoji = STATUS_EMOJI.get(status.lower().replace(" ", "_"), "📋")
            apps = sample_apps.get(status, [])
            st.markdown(
                f"<h4 style='color:{color}; font-size:0.9rem; font-weight:600;'>"
                f"{emoji} {status} ({len(apps)})</h4>",
                unsafe_allow_html=True,
            )
            for app in apps:
                st.markdown(
                    f"<div style='background:#1e1e2e; border:1px solid {color}40; border-radius:8px; "
                    f"padding:0.6rem 0.75rem; margin-bottom:0.4rem; font-size:0.85rem;'>{app}</div>",
                    unsafe_allow_html=True,
                )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Export CSV"):
            st.info("Exporting applications to CSV...")
    with col2:
        if st.button("📊 Export Excel"):
            st.info("Exporting applications to Excel...")


def page_cover_letters():
    st.title("📬 Cover Letters")
    st.markdown(
        "<p style='color:#94a3b8;'>AI-generated, personalised cover letters</p>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Generate New")
        job_title = st.text_input("Job Title", "Senior ML Engineer")
        company = st.text_input("Company", "Anthropic")
        tone = st.selectbox("Tone", ["Professional", "Enthusiastic", "Conversational"])
        if st.button("🤖 Generate Cover Letter", use_container_width=True):
            with st.spinner("Writing cover letter..."):
                time.sleep(2)
            st.success("Cover letter generated!")

    with col2:
        st.subheader("Generated Cover Letter")
        st.markdown(
            """<div style='background:#1e1e2e; border:1px solid #3f3f5a; border-radius:10px; 
               padding:1.5rem; font-size:0.9rem; line-height:1.7;'>

**Dear Hiring Manager,**

I'm writing to express my strong interest in the Senior ML Engineer role at Anthropic — 
a company whose mission to build reliable, interpretable AI aligns deeply with the work 
I've been doing for the past three years.

At TechCorp, I led the development of a RAG-based document intelligence system that reduced 
analyst research time by 60% and processed over 2M documents monthly. That project required 
the same blend of rigorous ML engineering and systems thinking that I know Anthropic values.

What draws me specifically to this role is the opportunity to work on safety-critical model 
infrastructure. My experience with RLHF pipelines and constitutional AI research positions 
me to contribute meaningfully from day one, while your team's frontier work would push me 
to grow in ways I can't achieve elsewhere.

I'd love to explore how my background aligns with your current priorities. I'm available 
for a conversation at your convenience.

Warmly,  
[Candidate Name]

</div>""",
            unsafe_allow_html=True,
        )
        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button("📥 Download .txt", "cover letter text here", "cover_letter.txt")
        with col_b:
            st.button("📋 Copy to Clipboard")


def page_analytics():
    st.title("📈 Analytics")
    st.markdown(
        "<p style='color:#94a3b8;'>Deep insights into your job search performance</p>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Match Scores Distribution")
        import numpy as np

        scores = np.random.normal(72, 15, 200).clip(0, 100)
        fig = px.histogram(
            x=scores,
            nbins=20,
            color_discrete_sequence=["#6366f1"],
            labels={"x": "Match Score", "y": "Count"},
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e2e8f0"},
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Jobs by Source")
        sources = pd.DataFrame(
            {
                "Source": ["LinkedIn", "Indeed", "Wellfound", "Company Sites"],
                "Count": [142, 67, 28, 10],
            }
        )
        fig = px.pie(
            sources,
            values="Count",
            names="Source",
            color_discrete_sequence=["#6366f1", "#8b5cf6", "#10b981", "#f59e0b"],
            hole=0.4,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e2e8f0"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Application Timeline")
    import numpy as np

    dates = pd.date_range("2025-01-01", periods=20, freq="3D")
    timeline = pd.DataFrame(
        {
            "Date": dates,
            "Applications": np.random.randint(1, 5, 20).cumsum(),
            "Interviews": np.random.randint(0, 2, 20).cumsum(),
        }
    )
    fig = px.line(
        timeline,
        x="Date",
        y=["Applications", "Interviews"],
        color_discrete_sequence=["#6366f1", "#10b981"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
        yaxis={"gridcolor": "#2a2a3e"},
        xaxis={"gridcolor": "#2a2a3e"},
    )
    st.plotly_chart(fig, use_container_width=True)


def page_settings():
    st.title("⚙️ Settings")

    with st.expander("🔑 API Configuration", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            provider = st.selectbox("LLM Provider", ["OpenAI", "Anthropic"])
            model = st.text_input("Model", value="gpt-4o")
        with col2:
            api_key = st.text_input("API Key", type="password", placeholder="sk-...")
            temp = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)

    with st.expander("📧 Email Notifications"):
        enable_email = st.toggle("Enable Email Notifications", value=False)
        if enable_email:
            st.text_input("Recipient Email")
            st.text_input("SendGrid API Key", type="password")

    with st.expander("🔎 Search Defaults"):
        st.text_input("Default Keywords", value="Python Engineer, ML Engineer")
        st.text_input("Default Location", value="Remote")
        st.slider("Default Min Match Score", 0, 100, 60)

    if st.button("💾 Save Settings", use_container_width=True):
        st.success("Settings saved!")


# ── Main ─────────────────────────────────────────────────────


def main():
    page = render_sidebar()

    if page == "🏠 Dashboard":
        page_dashboard()
    elif page == "🔍 Job Search":
        page_job_search()
    elif page == "📄 Resume":
        page_resume()
    elif page == "🎯 Matches":
        page_matches()
    elif page == "📊 Applications":
        page_applications()
    elif page == "📬 Cover Letters":
        page_cover_letters()
    elif page == "📈 Analytics":
        page_analytics()
    elif page == "⚙️ Settings":
        page_settings()


if __name__ == "__main__":
    main()
