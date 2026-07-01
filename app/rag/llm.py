"""LLM provider abstraction and answer generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from app.config import Config
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are TradeScope-AI, an educational trade and finance assistant.
Answer the user's question using ONLY the provided context from trusted web sources.
If the context does not contain enough information, say so clearly — do not invent facts.
Explain financial and trading terms in plain language when helpful.
When you use information from a source, mention the article title or URL.
Keep answers clear, concise, and well-structured.

DISCLAIMER: Your responses are for educational purposes only and are NOT financial advice."""


def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a context string for the LLM."""
    if not chunks:
        return "No relevant context was found."

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        title = chunk.get("title", "Unknown")
        url = chunk.get("source_url", "")
        section = chunk.get("section", "")
        text = chunk.get("text", "")
        header = f"[Source {i}] {title}"
        if section:
            header += f" — {section}"
        if url:
            header += f"\nURL: {url}"
        parts.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(parts)


class BaseLLMClient(ABC):
    """Abstract LLM client interface."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a completion from system and user prompts."""


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""

    def __init__(self):
        from openai import OpenAI

        kwargs = {"api_key": Config.API_KEY}
        if Config.LLM_BASE_URL:
            kwargs["base_url"] = Config.LLM_BASE_URL
        self.client = OpenAI(**kwargs)
        self.model = Config.LLM_MODEL_NAME

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=Config.LLM_MAX_TOKENS,
            temperature=Config.LLM_TEMPERATURE,
        )
        return response.choices[0].message.content or ""


class AnthropicClient(BaseLLMClient):
    """Anthropic API client."""

    def __init__(self):
        from anthropic import Anthropic

        self.client = Anthropic(api_key=Config.API_KEY)
        self.model = Config.LLM_MODEL_NAME

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=Config.LLM_MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=Config.LLM_TEMPERATURE,
        )
        text_parts = [block.text for block in response.content if block.type == "text"]
        return "".join(text_parts)


class LocalLLMClient(BaseLLMClient):
    """
    Local LLM via OpenAI-compatible API (Ollama, LM Studio, vLLM, etc.).

    Set LLM_BASE_URL (e.g. http://localhost:11434/v1) and LLM_MODEL_NAME.
  """

    def __init__(self):
        from openai import OpenAI

        base_url = Config.LLM_BASE_URL or "http://localhost:11434/v1"
        self.client = OpenAI(api_key=Config.API_KEY or "not-needed", base_url=base_url)
        self.model = Config.LLM_MODEL_NAME

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=Config.LLM_MAX_TOKENS,
            temperature=Config.LLM_TEMPERATURE,
        )
        return response.choices[0].message.content or ""


_llm_client: Optional[BaseLLMClient] = None


def get_llm_client(provider: Optional[str] = None) -> BaseLLMClient:
    """Return a cached LLM client for the configured provider."""
    global _llm_client
    provider = (provider or Config.LLM_PROVIDER).lower()

    if _llm_client is None:
        if provider == "openai":
            Config.validate_llm_config()
            _llm_client = OpenAIClient()
        elif provider == "anthropic":
            Config.validate_llm_config()
            _llm_client = AnthropicClient()
        elif provider == "local":
            _llm_client = LocalLLMClient()
        else:
            raise ValueError(
                f"Unsupported LLM_PROVIDER: {provider}. "
                "Use 'openai', 'anthropic', or 'local'."
            )
        logger.info("Initialized LLM client: %s (%s)", provider, Config.LLM_MODEL_NAME)

    return _llm_client


def generate_answer(
    query: str,
    context_chunks: List[Dict[str, Any]],
    llm_client: Optional[BaseLLMClient] = None,
) -> str:
    """
    Generate an answer using retrieved context and the configured LLM.

    Args:
        query: User question.
        context_chunks: Retrieved document chunks with metadata.
        llm_client: Optional LLM client override.

    Returns:
        Generated answer string.
    """
    client = llm_client or get_llm_client()
    context = build_context_block(context_chunks)

    user_prompt = f"""Context from knowledge base:
{context}

User question: {query}

Answer the question based on the context above. Cite sources when relevant."""

    try:
        return client.generate(SYSTEM_PROMPT, user_prompt).strip()
    except Exception as exc:
        logger.exception("LLM generation failed")
        raise RuntimeError(f"Failed to generate answer: {exc}") from exc
