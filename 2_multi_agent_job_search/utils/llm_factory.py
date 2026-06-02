"""
utils/llm_factory.py
─────────────────────
Factory that returns the correct LangChain ChatModel based on LLM_PROVIDER.
Single place to swap models — no provider imports scattered across agents.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.language_models import BaseChatModel
from loguru import logger

from config import settings


@lru_cache(maxsize=4)
def get_llm(
    temperature: float | None = None,
    model: str | None = None,
    streaming: bool = False,
) -> BaseChatModel:
    """
    Return a cached ChatModel instance.

    Args:
        temperature: Override settings.llm_temperature.
        model:       Override settings.llm_model.
        streaming:   Enable token streaming.
    """
    temp = temperature if temperature is not None else settings.llm_temperature
    mdl = model or settings.llm_model

    kwargs: dict[str, Any] = {
        "temperature": temp,
        "streaming": streaming,
    }

    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=mdl,
            api_key=settings.openai_api_key,
            max_tokens=settings.llm_max_tokens,
            **kwargs,
        )
        logger.debug(f"LLM: OpenAI {mdl} (temp={temp})")
        return llm

    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(
            model=mdl or "claude-sonnet-4-20250514",
            api_key=settings.anthropic_api_key,
            max_tokens=settings.llm_max_tokens,
            **kwargs,
        )
        logger.debug(f"LLM: Anthropic {mdl} (temp={temp})")
        return llm

    raise ValueError(f"Unknown LLM provider: {settings.llm_provider!r}")


def get_embedding_model():
    """Return a LangChain embeddings instance."""
    if settings.llm_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            api_key=settings.openai_api_key,
        )

    # Anthropic doesn't have native embeddings — fall back to OpenAI
    from langchain_openai import OpenAIEmbeddings

    logger.warning("Anthropic selected but using OpenAI embeddings (no Anthropic embedding API).")
    return OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=settings.openai_api_key,
    )
