# 🤖 Multi-Agent Job Search Assistant

> An autonomous AI system that searches, matches, and applies to jobs — so you can focus on interviews.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![CrewAI](https://img.shields.io/badge/CrewAI-0.80-purple)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit Dashboard                         │
│   Resume Upload │ Job Search │ Matches │ Tracker │ Analytics    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   CrewAI Crew   │
                    │  Orchestrator   │
                    └────────┬────────┘
           ┌─────────────────┼─────────────────────────┐
           │                 │                         │
  ┌────────▼───────┐ ┌───────▼────────┐ ┌─────────────▼──────────┐
  │ Job Research   │ │ Resume Analysis│ │   Job Matching Agent   │
  │ Agent          │ │ Agent          │ │ (Vector + LLM Scoring) │
  │                │ │                │ │                        │
  │ • LinkedIn     │ │ • PDF/DOCX     │ │ • 0-100 Score          │
  │ • Indeed       │ │   Parser       │ │ • Skill gap analysis   │
  │ • Wellfound    │ │ • LLM Struct.  │ │ • Job ranking          │
  │ • Company pages│ │ • FAISS/Chroma │ │                        │
  └────────┬───────┘ └───────┬────────┘ └─────────────┬──────────┘
           │                 │                         │
  ┌────────▼───────┐ ┌───────▼────────┐ ┌─────────────▼──────────┐
  │Resume Custom.  │ │ Cover Letter   │ │ Application Tracking   │
  │Agent           │ │ Agent          │ │ Agent                  │
  │                │ │                │ │                        │
  │ • ATS keywords │ │ • Company      │ │ • SQLite/PostgreSQL     │
  │ • Bullet rewrit│ │   research     │ │ • Status tracking       │
  │ • Role-specific│ │ • Personalised │ │ • CSV/Excel export     │
  └────────────────┘ └────────────────┘ └─────────────┬──────────┘
                                                       │
                                          ┌────────────▼───────────┐
                                          │  Notification Agent    │
                                          │ • Email alerts         │
                                          │ • Weekly reports       │
                                          │ • Status updates       │
                                          └────────────────────────┘
```

### Data Flow

```
Resume PDF/DOCX
    │
    ▼
Text Extraction (pdfplumber / python-docx)
    │
    ▼
LLM Structuring → {skills, experience, education, projects}
    │
    ▼
Embedding → FAISS / ChromaDB
    │
    ◄──────────── Job Descriptions (LinkedIn / Indeed / Wellfound)
    │
    ▼
Cosine Similarity + LLM Scoring → Match Score (0-100)
    │
    ▼
Resume Customisation → ATS-optimised tailored resume
    │
    ▼
Cover Letter Generation → Personalised, company-specific
    │
    ▼
Database Persistence → Track applications, generate reports
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- OpenAI or Anthropic API key
- (Optional) SendGrid key for email notifications

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/multi-agent-job-search.git
cd multi-agent-job-search
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Initialise Database

```bash
python main.py init-db
```

### 4. Launch Dashboard

```bash
streamlit run frontend/app.py
# → http://localhost:8501
```

---

## 🐳 Docker

```bash
# Copy and configure .env
cp .env.example .env

# Start all services
docker-compose up -d

# Dashboard available at http://localhost:8501
```

---

## 💻 CLI Usage

```bash
# Full end-to-end job search
python main.py search \
  --keywords "Senior ML Engineer, Python Engineer" \
  --location Remote \
  --resume ./data/resumes/my_resume.pdf \
  --min-score 70

# Match existing resume to saved jobs
python main.py match --resume ./data/resumes/my_resume.pdf

# Generate cover letter + customised resume for one job
python main.py apply \
  --resume ./data/resumes/my_resume.pdf \
  --job-title "Senior ML Engineer" \
  --company "Anthropic" \
  --job-desc ./data/jobs/anthropic_jd.txt

# Generate weekly report
python main.py report

# Launch dashboard
python main.py dashboard
```

---

## 🏗️ Project Structure

```
multi_agent_job_search/
│
├── agents/                         # CrewAI agent definitions
│   ├── job_research_agent.py       # LinkedIn, Indeed, Wellfound scraper
│   ├── resume_analysis_agent.py    # PDF/DOCX parser + embedder
│   ├── job_matching_agent.py       # Semantic scoring engine
│   ├── resume_customization_agent.py # ATS optimiser
│   ├── cover_letter_agent.py       # Personalised writing
│   ├── application_tracking_agent.py # DB + CSV tracker
│   └── notification_agent.py       # Email alerts + reports
│
├── tasks/
│   └── job_search_tasks.py         # CrewAI Task definitions
│
├── crews/
│   └── workflows.py                # Crew orchestration (3 workflows)
│
├── tools/                          # CrewAI @tool functions
│   ├── scraping_tools.py           # Web scraping (Playwright + BS4)
│   ├── resume_tools.py             # Parse, embed, customise resumes
│   ├── matching_tools.py           # Scoring + ranking
│   ├── writing_tools.py            # Cover letters + reports
│   └── notification_tools.py       # Email (SendGrid)
│
├── database/
│   ├── models.py                   # SQLAlchemy ORM models
│   ├── connection.py               # Async engine + sessions
│   └── repository.py              # CRUD repository layer
│
├── vectorstore/
│   └── store.py                    # FAISS / ChromaDB abstraction
│
├── config/
│   └── settings.py                 # Pydantic-v2 settings
│
├── utils/
│   ├── llm_factory.py              # Multi-provider LLM factory
│   └── logger.py                   # Loguru setup
│
├── frontend/
│   └── app.py                      # Streamlit dashboard (8 pages)
│
├── tests/
│   └── test_matching.py            # Unit + async tests
│
├── data/
│   ├── resumes/                    # Uploaded resumes
│   ├── jobs/                       # Cached job data
│   └── reports/                    # Generated CSV/Excel reports
│
├── main.py                         # CLI entry point (Click)
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🤖 Agents Deep Dive

### 1. Job Research Agent
- **Tools**: `search_linkedin_jobs`, `search_indeed_jobs`, `search_wellfound_jobs`, `scrape_company_jobs`
- **Strategy**: Playwright for JS-heavy sites, httpx+BS4 for static pages
- **Deduplication**: MD5 hash of (title + company) + semantic similarity check
- **Rate limiting**: Random delay 2-5s between requests, respectful scraping

### 2. Resume Analysis Agent
- **Tools**: `parse_resume`, `embed_resume`
- **Parsing**: pdfplumber for PDFs, python-docx for DOCX, fallback to OCR
- **Structuring**: LLM extraction of skills, experience, education, projects
- **Embedding**: OpenAI `text-embedding-3-large` (3072 dims) → FAISS

### 3. Job Matching Agent
- **Tools**: `compute_match_score`, `rank_jobs_by_match`, `identify_skill_gaps`
- **Scoring**: Two-pass — fast vector similarity, then LLM analysis for top-N
- **Output**: Score 0-100 with breakdown (technical skills 40%, experience 25%, domain 20%, education 10%, soft skills 5%)

### 4. Resume Customization Agent
- **Tools**: `customize_resume`
- **Approach**: Keyword injection, bullet point strengthening, section reordering
- **Constraint**: Never fabricates experience — only highlights and reframes

### 5. Cover Letter Agent
- **Tools**: `generate_cover_letter`, `research_company`, `generate_interview_prep`
- **Structure**: Hook → Achievement → Fit → Call to action
- **Also generates**: 5 likely interview questions with STAR answer frameworks

### 6. Application Tracking Agent
- **Tools**: `save_jobs_to_database`, `create_application`, `update_application_status`, `get_application_statistics`, `export_applications_csv`
- **Tracks**: discovered → applied → phone screen → interview → final round → offer/rejected

### 7. Notification Agent
- **Tools**: `send_email_notification`, `send_new_jobs_alert`, `track_application_status_change`, `generate_weekly_report`
- **Triggers**: New high-match jobs, status changes, weekly Sunday analytics

---

## 🔧 Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` or `anthropic` |
| `LLM_MODEL` | `gpt-4o` | Model name |
| `VECTOR_STORE_TYPE` | `faiss` | `faiss` or `chromadb` |
| `MAX_JOBS_PER_SOURCE` | `50` | Cap per scraping run |
| `SCRAPE_DELAY_MIN` | `2.0` | Min delay between requests (s) |
| `ENABLE_EMAIL_NOTIFICATIONS` | `false` | Requires SendGrid key |

---

## 🧪 Testing

```bash
pytest tests/ -v                    # All tests
pytest tests/ -v -k "match"         # Match tests only
pytest tests/ --asyncio-mode=auto   # With async tests
```

---

## 📊 Database Schema

| Table | Purpose |
|---|---|
| `resumes` | Uploaded and parsed resumes |
| `jobs` | Discovered job listings |
| `match_scores` | Resume × job match analysis |
| `applications` | Application lifecycle tracking |
| `notifications` | Email send log |
| `search_history` | Audit trail of search runs |

---

## 🔒 Ethical Scraping

This project scrapes job boards for personal job searching. Please:
- Respect `robots.txt` for production use
- Use rate limiting (built-in 2-5s delays)
- Do not sell or redistribute scraped data
- Check each platform's ToS before commercial use

---

## 📄 License

MIT — see [LICENSE](LICENSE)

---

## 🙏 Acknowledgements

Built with [CrewAI](https://crewai.com), [LangChain](https://langchain.com), 
[Streamlit](https://streamlit.io), and the OpenAI / Anthropic APIs.
