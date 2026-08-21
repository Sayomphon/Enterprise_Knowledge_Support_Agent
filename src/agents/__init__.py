"""Shared LLM client construction for the agent boundaries.

Only modules in this package may call an LLM (AGENTS.md section 3);
everything else in ``src`` stays off the network by construction. This
module is also the single place where a missing credential is detected:
the check happens here, at the boundary, rather than at process start, so
guardrail refusals and low-score fallbacks -- which never reach an LLM --
keep working without any key (remediation plan Finding 8).
"""

from functools import lru_cache

from langchain_openai import ChatOpenAI

from src.config import (
    LLM_MAX_RETRIES,
    LLM_TIMEOUT_SECONDS,
    MODEL_NAME,
    TEMPERATURE,
    has_llm_credential,
)


class MissingLlmCredentialError(RuntimeError):
    """Raised when an LLM path is reached without configured credentials."""


def get_llm(model_name: str | None = None) -> ChatOpenAI:
    """Return the client for one model, refusing to build it unconfigured.

    The credential check sits outside the cache on purpose: a cached
    client would otherwise outlive the credential it was built with, and
    the boundary would stop reporting the environment as it is now.

    Args:
        model_name: Explicit model override; defaults to the configured
            ``MODEL_NAME``.

    Returns:
        A cached ``ChatOpenAI`` client for the resolved model name.

    Raises:
        MissingLlmCredentialError: If no credential is configured. The
            message names the variable, never a value, and the graph maps
            the exception to the ``llm_not_configured`` reason code so the
            employee is told the service is unavailable rather than that
            the corpus lacked evidence.
    """
    if not has_llm_credential():
        raise MissingLlmCredentialError(
            "OPENAI_API_KEY is not configured; LLM routes are unavailable"
        )
    return _build_llm(model_name or MODEL_NAME)


@lru_cache(maxsize=8)
def _build_llm(model_name: str) -> ChatOpenAI:
    """Build one ChatOpenAI client per distinct model name.

    Every client carries an explicit request timeout and a bounded retry
    budget so a stalled provider cannot hang a request indefinitely; the
    SDK retries only transient provider errors. gpt-5 family models accept
    only the default temperature, so the parameter is passed to other
    models only.

    Args:
        model_name: Resolved model name; never ``None`` at this point.

    Returns:
        A cached ``ChatOpenAI`` client for that model name.
    """
    if model_name.startswith("gpt-5"):
        return ChatOpenAI(
            model=model_name,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=LLM_MAX_RETRIES,
        )
    return ChatOpenAI(
        model=model_name,
        temperature=TEMPERATURE,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
    )
