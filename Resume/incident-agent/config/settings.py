from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):

    # ── Model configuration ──
    coordinator_model: str = Field(default="qwen2.5:3b")
    knowledge_model: str = Field(default="qwen2.5:3b")
    resolution_model: str = Field(default="gemma3:4b")
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")

    # ── Ollama ──
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_temperature: float = Field(default=0.1)
    ollama_timeout: int = Field(default=120)

    # ── ChromaDB ──
    chroma_persist_dir: str = Field(
        default=str(BASE_DIR / "knowledge_base" / "chroma_store")
    )
    chroma_collection_name: str = Field(default="incident_knowledge")

    # ── Agent behaviour ──
    max_retries: int = Field(default=3)
    confidence_threshold: float = Field(default=0.7)
    max_search_results: int = Field(default=5)
    max_iterations: int = Field(default=10)

    # ── Memory ──
    session_memory_limit: int = Field(default=20)
    historical_memory_db: str = Field(
        default=str(BASE_DIR / "data" / "incident_history.db")
    )

    # ── Logging ──
    log_level: str = Field(default="INFO")
    log_file: str = Field(default=str(BASE_DIR / "logs" / "agent.log"))

    # ── MCP ──
    mcp_server_host: str = Field(default="localhost")
    mcp_server_port: int = Field(default=8000)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Single instance imported everywhere
settings = Settings()