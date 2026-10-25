"""Single place where settings are read. Import `settings` everywhere else."""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()  # reads .env from the project root; real env vars win


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    judge_provider: str = os.getenv("JUDGE_PROVIDER", "gemini")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "")
    judge_ollama_model: str = os.getenv("JUDGE_OLLAMA_MODEL", "llama3.1:8b")
    database_url: str = os.getenv("DATABASE_URL", "postgresql://triage:triage@localhost:5432/triage")
    fhir_base_url: str = os.getenv("FHIR_BASE_URL", "http://localhost:8080/fhir")
    chroma_dir: str = os.getenv("CHROMA_DIR", "./data/chroma")
    mcp_host: str = os.getenv("MCP_HOST", "127.0.0.1")
    mcp_port: int = int(os.getenv("MCP_PORT", "8765"))
    critic_host: str = os.getenv("CRITIC_HOST", "127.0.0.1")
    critic_port: int = int(os.getenv("CRITIC_PORT", "9105"))


settings = Settings()