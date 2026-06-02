"""
config/settings.py
──────────────────
Centralised Pydantic-v2 settings loaded from .env.
All modules import `settings` from here — never read os.environ directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ─────────────────────────────────────────────────
    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    llm_provider: Literal["openai", "anthropic"] = "openai"
    llm_model: str = "gpt-4o"
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=4096, ge=256)

    # ── Embeddings ──────────────────────────────────────────
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072

    # ── Database ────────────────────────────────────────────
    database_url: str = "sqlite:///./data/jobsearch.db"

    # ── Vector Store ────────────────────────────────────────
    vector_store_type: Literal["faiss", "chromadb"] = "faiss"
    vector_store_path: str = "./vectorstore/faiss_index"
    chroma_persist_dir: str = "./vectorstore/chroma"

    # ── Scraping ────────────────────────────────────────────
    use_headless_browser: bool = True
    playwright_browser: str = "chromium"
    scrape_delay_min: float = 2.0
    scrape_delay_max: float = 5.0
    max_jobs_per_source: int = 50

    # ── Job Search ──────────────────────────────────────────
    default_search_keywords: str = "Software Engineer,Python Developer"
    default_location: str = "Remote"
    salary_currency: str = "USD"

    # ── Email ────────────────────────────────────────────────
    sendgrid_api_key: str = ""
    notification_email_from: str = ""
    notification_email_to: str = ""
    enable_email_notifications: bool = False

    # ── App ──────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_file: str = "./data/logs/app.log"
    max_retries: int = 3
    request_timeout: int = 30
    batch_size: int = 10

    # ── CrewAI ───────────────────────────────────────────────
    crew_verbose: bool = True
    crew_max_iter: int = 15
    crew_memory: bool = True

    # ── Reports ──────────────────────────────────────────────
    report_output_dir: str = "./data/reports"

    # ── Computed helpers ────────────────────────────────────
    @property
    def active_api_key(self) -> str:
        if self.llm_provider == "openai":
            return self.openai_api_key
        return self.anthropic_api_key

    @property
    def search_keywords_list(self) -> list[str]:
        return [k.strip() for k in self.default_search_keywords.split(",")]

    @field_validator("log_file", "vector_store_path", "chroma_persist_dir", "report_output_dir", mode="before")
    @classmethod
    def _ensure_parent_exists(cls, v: str) -> str:
        Path(v).parent.mkdir(parents=True, exist_ok=True)
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Module-level singleton — import this everywhere
settings = get_settings()
