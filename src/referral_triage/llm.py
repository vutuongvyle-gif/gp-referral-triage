from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama, OllamaEmbeddings

from referral_triage.config import settings


def get_llm(role: str = "agent") -> BaseChatModel:
    """role='agent' for the triage agents, role='judge' for DeepEval."""
    provider = settings.llm_provider if role == "agent" else settings.judge_provider

    if provider == "ollama":
        model = settings.ollama_model if role == "agent" else settings.judge_ollama_model
        return ChatOllama(
            model=model,
            base_url=settings.ollama_base_url,
            temperature=0,      # deterministic-ish output for evaluation
            num_ctx=8192,       # Ollama's default context is small; referrals are long
            reasoning=False,    # turn off qwen3 "thinking" for speed and clean JSON
        )
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        if not settings.gemini_model:
            raise RuntimeError("Set GEMINI_MODEL in .env (copy a current ID from AI Studio)")
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model, api_key=settings.gemini_api_key, temperature=0
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-haiku-5-5", temperature=0)
    raise ValueError(f"Unknown provider: {provider}")


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=settings.ollama_embed_model, base_url=settings.ollama_base_url)