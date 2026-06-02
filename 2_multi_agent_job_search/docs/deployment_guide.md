# Deployment Guide

## Local Development

```bash
git clone <repo>
cd multi_agent_job_search
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
cp .env.example .env       # Fill in API keys
python main.py init-db
streamlit run frontend/app.py
```

## Docker (Recommended for Production)

```bash
cp .env.example .env       # Fill in API keys
docker-compose up -d       # Starts app + PostgreSQL
# Dashboard: http://localhost:8501
docker-compose logs -f app # Follow logs
docker-compose down        # Stop
```

## AWS EC2 (Minimal)

```bash
# Launch Ubuntu 22.04 t3.medium (2 vCPU, 4 GB RAM minimum)
sudo apt update && sudo apt install -y docker.io docker-compose
git clone <repo> && cd multi_agent_job_search
cp .env.example .env && nano .env
sudo docker-compose up -d
# Configure security group: inbound TCP 8501
```

## Environment Variables Checklist

- [ ] `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`
- [ ] `DATABASE_URL` (SQLite OK for single user, PostgreSQL for multi-user)
- [ ] `VECTOR_STORE_TYPE` (faiss for local, chromadb for networked)
- [ ] `SENDGRID_API_KEY` + email vars (optional, for notifications)
- [ ] `ENABLE_EMAIL_NOTIFICATIONS=true` (only after configuring SendGrid)

## Database Migrations

```bash
# Using Alembic (after adding new models)
alembic init alembic
alembic revision --autogenerate -m "add new table"
alembic upgrade head
```

## Monitoring

- Logs: `./data/logs/app.log` (JSON, rotated at 10MB, kept 14 days)
- Metrics: Streamlit Analytics page shows application pipeline stats
- Errors: All agent failures are logged with full tracebacks via Loguru
