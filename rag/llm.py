"""LLM module: generate answers via a local Ollama model.

Uses ``langchain-ollama`` to communicate with Ollama running on the
host (or in a Docker sibling container).  The module provides a simple
``generate_answer`` function that accepts a pre-built prompt and returns
the LLM's text response.
"""

from __future__ import annotations

import logging

from langchain_ollama import OllamaLLM

from rag.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TEMPERATURE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton LLM instance
# ---------------------------------------------------------------------------
_llm: OllamaLLM | None = None


def get_llm() -> OllamaLLM:
    """Return a cached ``OllamaLLM`` instance.

    The instance is created once and reused for subsequent calls.

    Returns:
        OllamaLLM: Configured LangChain Ollama wrapper.
    """
    global _llm  # noqa: PLW0603
    if _llm is None:
        logger.info(
            "Connecting to Ollama at %s with model %s",
            OLLAMA_BASE_URL,
            OLLAMA_MODEL,
        )
        _llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=OLLAMA_TEMPERATURE,
        )
    return _llm


def is_ollama_available() -> bool:
    """Check whether the Ollama server is reachable and the model is loaded.

    Returns:
        True if a test prompt succeeds, False otherwise.
    """
    try:
        llm = get_llm()
        llm.invoke("Hi")
        return True
    except Exception:
        logger.warning("Ollama is not reachable at %s", OLLAMA_BASE_URL)
        return False


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

_FALLBACK_MESSAGE = (
    "⚠️ **Ollama is not running or the model is not available.**\n\n"
    "Please make sure:\n"
    "1. Ollama is installed → https://ollama.com\n"
    "2. The Ollama server is running → `ollama serve`\n"
    "3. The model is pulled → `ollama pull {model}`\n\n"
    "You can change the model or URL in `.env`."
)


def generate_answer(prompt: str) -> str:
    """Send *prompt* to Ollama and return the generated text.

    If Ollama is unreachable or the model is missing, a friendly
    fallback message is returned instead of raising an exception.

    Args:
        prompt: The fully assembled RAG prompt (context + question).

    Returns:
        The LLM's answer as a plain string.
    """
    try:
        llm = get_llm()
        response: str = llm.invoke(prompt)
        return response.strip()
    except ConnectionError:
        logger.error("Connection refused — is Ollama running?")
        return _FALLBACK_MESSAGE.format(model=OLLAMA_MODEL)
    except Exception as exc:
        error_msg = str(exc).lower()
        if "connection" in error_msg or "refused" in error_msg:
            return _FALLBACK_MESSAGE.format(model=OLLAMA_MODEL)
        logger.exception("LLM generation failed")
        return (
            f"❌ **LLM Error:** {exc}\n\n"
            "Check that Ollama is running and the model is available."
        )
