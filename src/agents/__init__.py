"""Shared LLM client construction for the agent boundaries.

Only modules in this package may call an LLM (AGENTS.md section 3);
everything else in ``src`` stays off the network by construction. This
module is also the single place where a missing credential is detected:
the check happens here, at the boundary, rather than at process start, so
guardrail refusals and low-score fallbacks -- which never reach an LLM --
keep working without any key (remediation plan Finding 8).
"""

import hashlib
import math
import os
from functools import lru_cache

from langchain_openai import ChatOpenAI

from src.config import (
    LLM_MAX_RETRIES,
    LLM_REWRITE_MAX_RETRIES,
    LLM_REWRITE_TIMEOUT_SECONDS,
    LLM_TIMEOUT_SECONDS,
    MODEL_NAME,
    TEMPERATURE,
    has_llm_credential,
)


class MissingLlmCredentialError(RuntimeError):
    """Raised when an LLM path is reached without configured credentials."""


def get_llm(
    model_name: str | None = None, *, budget_seconds: float | None = None
) -> ChatOpenAI:
    """Return the client for one model, refusing to build it unconfigured.

    The credential check sits outside the cache, and a fingerprint of the
    credential is part of the cache key. The check alone only detects a
    credential being REMOVED: a rotated or revoked key leaves
    ``has_llm_credential`` true while the cache keeps handing back a
    client built with the old secret, so every request fails against a
    dead key and degrades as ``reporter_failure`` -- never as
    ``llm_not_configured`` -- until the process restarts. Keying on the
    fingerprint retires that client the moment the value changes.

    Args:
        model_name: Explicit model override; defaults to the configured
            ``MODEL_NAME``.
        budget_seconds: What is left of the request deadline. ``None``
            leaves the configured boundary timeout untouched, which is
            what a caller outside the graph gets.

    Returns:
        A cached ``ChatOpenAI`` client for the resolved model name, the
        credential currently in the environment, and the timeout that
        budget allows.

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
    return _build_llm(
        model_name or MODEL_NAME,
        credential_fingerprint(),
        _effective_timeout(LLM_TIMEOUT_SECONDS, budget_seconds),
        LLM_MAX_RETRIES,
    )


def get_rewrite_llm(
    model_name: str | None = None, *, budget_seconds: float | None = None
) -> ChatOpenAI:
    """Return the client for the rewrite boundary, on a tighter budget.

    Args:
        model_name: Explicit model override; defaults to ``MODEL_NAME``.
        budget_seconds: What is left of the request deadline; see
            ``get_llm``.

    Returns:
        A cached client carrying ``LLM_REWRITE_TIMEOUT_SECONDS`` and
        ``LLM_REWRITE_MAX_RETRIES`` instead of the shared budget, because
        this boundary sits in front of the reporter on the same request
        and the two waits add up for the employee. The timeout is trimmed
        further when the request has less than that left.

    Raises:
        MissingLlmCredentialError: If no credential is configured.
    """
    if not has_llm_credential():
        raise MissingLlmCredentialError(
            "OPENAI_API_KEY is not configured; LLM routes are unavailable"
        )
    return _build_llm(
        model_name or MODEL_NAME,
        credential_fingerprint(),
        _effective_timeout(LLM_REWRITE_TIMEOUT_SECONDS, budget_seconds),
        LLM_REWRITE_MAX_RETRIES,
    )


def _effective_timeout(
    configured: float, budget_seconds: float | None
) -> float:
    """Reduce a boundary timeout to what the request has left.

    Args:
        configured: The boundary's own configured timeout.
        budget_seconds: Remaining request budget, or ``None`` when the
            caller tracks no deadline.

    Returns:
        The smaller of the two, floored to whole seconds so that requests
        differing by milliseconds share one cached client instead of each
        building their own, and never below one second: a sub-second
        timeout would fail every call it was applied to, and the graph
        skips a boundary whose budget is already gone rather than
        starting it with an impossible one.
    """
    if budget_seconds is None:
        return configured
    return max(1.0, min(configured, float(math.floor(budget_seconds))))


def credential_fingerprint() -> str:
    """Identify the configured credential without exposing it.

    Secrets never leave ``config`` as values (AGENTS.md section 7), so the
    cache key is a truncated digest rather than the key itself: it is
    enough to tell one credential from another and carries nothing that
    could reach a log, a state field, or a traceback.

    Returns:
        A short hex digest of the configured credential, or ``""`` when
        none is set.
    """
    secret = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not secret:
        return ""
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()[:16]


# One entry per (model, credential, timeout, retries). The timeout is now
# request-dependent, so the cache holds a few clients per boundary rather
# than one; the bound stays small because timeouts are whole seconds.
@lru_cache(maxsize=32)
def _build_llm(
    model_name: str,
    credential_id: str,
    timeout_seconds: float,
    max_retries: int,
) -> ChatOpenAI:
    """Build one ChatOpenAI client per model name and credential.

    Every client carries an explicit request timeout and a bounded retry
    budget so a stalled provider cannot hang a request indefinitely; the
    SDK retries only transient provider errors. gpt-5 family models accept
    only the default temperature, so the parameter is passed to other
    models only.

    Args:
        model_name: Resolved model name; never ``None`` at this point.
        credential_id: Digest of the credential this client will
            be built with. It is never read here -- the SDK reads the
            environment itself -- and exists only to make the cache entry
            expire when the credential changes.
        timeout_seconds: Per-request timeout for this boundary.
        max_retries: Retry budget for this boundary.

    Returns:
        A cached ``ChatOpenAI`` client for that pair.
    """
    if model_name.startswith("gpt-5"):
        return ChatOpenAI(
            model=model_name,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )
    return ChatOpenAI(
        model=model_name,
        temperature=TEMPERATURE,
        timeout=timeout_seconds,
        max_retries=max_retries,
    )
