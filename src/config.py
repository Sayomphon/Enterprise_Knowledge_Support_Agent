"""Typed runtime settings loaded once from the environment.

Single source of configuration for the pipeline (AGENTS.md section 3).
This module imports nothing from the rest of ``src`` and validates every
value at import time, so a misconfigured deployment fails before any
corpus loading, retrieval, or LLM call can run on bad settings.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_str(name: str, default: str) -> str:
    """Read one setting, treating unset or blank values as the default.

    A blank line such as ``TOP_K=`` in ``.env`` must select the shared
    default rather than an empty value, so every setting reads through
    this single rule.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset or blank.

    Returns:
        The stripped environment value, or the default.
    """
    return (os.getenv(name) or "").strip() or default


def _env_float(name: str, default: str) -> float:
    """Read one float setting, naming the variable on malformed input.

    Args:
        name: Environment variable name.
        default: Fallback literal parsed when the variable is unset.

    Returns:
        The parsed float value.

    Raises:
        ValueError: If the resolved value is not a valid float.
    """
    raw = _env_str(name, default)
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None


def _env_int(name: str, default: str) -> int:
    """Read one integer setting, naming the variable on malformed input.

    Args:
        name: Environment variable name.
        default: Fallback literal parsed when the variable is unset.

    Returns:
        The parsed integer value.

    Raises:
        ValueError: If the resolved value is not a valid integer.
    """
    raw = _env_str(name, default)
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


def _env_flag(name: str, default: str) -> bool:
    """Read one boolean setting, naming the variable on unknown input.

    Feature flags must fail loudly rather than quietly resolving a typo
    such as ``ENABLE_OPS_VIEW=treu`` to the permissive value.

    Args:
        name: Environment variable name.
        default: Fallback literal parsed when the variable is unset.

    Returns:
        The parsed boolean value.

    Raises:
        ValueError: If the resolved value is not a recognised literal.
    """
    raw = _env_str(name, default).lower()
    if raw in {"true", "1", "yes", "on"}:
        return True
    if raw in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {raw!r}")


MODEL_NAME: str = _env_str("MODEL_NAME", "gpt-5-mini")
TEMPERATURE: float = _env_float("TEMPERATURE", "0")
# Client hardening: a stalled provider must never hang a request forever,
# and the SDK retries only transient provider errors.
LLM_TIMEOUT_SECONDS: float = _env_float("LLM_TIMEOUT_SECONDS", "30")
LLM_MAX_RETRIES: int = _env_int("LLM_MAX_RETRIES", "2")
# The rewrite boundary gets its own, tighter budget. The two LLM calls of
# a medium-band request are sequential and ``graph.invoke`` is
# synchronous, so their budgets add up in front of the employee: at the
# shared 30s x 3 attempts they reach 180 seconds of frozen UI before the
# fallback text appears. The rewrite is the optional half of that -- it
# improves recall, it does not produce the answer -- so it is the half
# that is cut. Worst case becomes 10s x 2 + 30s x 3 = 110s. That is still
# a long wait; a per-request deadline shared across both boundaries is
# the real fix and is not implemented here.
LLM_REWRITE_TIMEOUT_SECONDS: float = _env_float(
    "LLM_REWRITE_TIMEOUT_SECONDS", "10"
)
LLM_REWRITE_MAX_RETRIES: int = _env_int("LLM_REWRITE_MAX_RETRIES", "1")

# Calibrated 2026-08-20 against eval/retrieval_calibration.json with the
# character (2,5) TF-IDF configuration; the (2,4)/(2,5)/(3,5) ablation all
# reached Hit@3 12/12, and (2,5) gave the widest low-band gap. Raw top-1
# scores: generic OOD <= 0.0858 vs weakest answerable 0.1187, so the floor
# sits between them. The salary hard negative scored 0.1773 raw vs 0.1972
# for the next answerable case, so the direct threshold stays above it.
REWRITE_FLOOR: float = _env_float("REWRITE_FLOOR", "0.10")
DIRECT_ANSWER_THRESHOLD: float = _env_float("DIRECT_ANSWER_THRESHOLD", "0.19")
# Recalibrated 2026-08-20 on the 21-case set after the supported-scope gate
# landed. The first calibration had to hold this threshold at 0.24 because
# expanded scores of the salary hard negative (0.2152) and the weakest
# answerable slang case (0.2176) were not separable, which cost that slang
# case an answer. The hard negative is now stopped deterministically by
# scope validation before the rewrite branch, so the threshold no longer
# has to arbitrate between them: at 0.21 every labelled-answerable case
# routes to "answered" (coverage 12/12) with precision still 12/12, and
# 0.21 keeps a margin above the highest expanded score observed for an
# unsupported in-domain case (0.1884). The trade-off is explicit --
# precision on this band now rests on the scope gate rather than on the
# score alone, so weakening the scope catalog would weaken this threshold
# too.
FINAL_ANSWER_THRESHOLD: float = _env_float("FINAL_ANSWER_THRESHOLD", "0.21")

# Scope-gate alias similarity, calibrated 2026-08-20 against the 21-case
# eval/retrieval_calibration.json with character (2,4) containment. Every
# answerable case scores 0.5714 or above -- the weakest is the "lapakron"
# typo, while exact and slang aliases score 1.0000. Among the cases that
# must not resolve to a topic on their supported aliases, the strongest
# is 0.1667 (the under-specified "how many days of leave"), followed by
# 0.1429 and 0.1333 for out-of-domain queries; the in-domain hard
# negatives are rejected by their unsupported alias instead. 0.40 sits
# near the midpoint of that 0.1667-0.5714 gap. Held-out data was not
# consulted for this number.
SCOPE_MATCH_THRESHOLD: float = _env_float("SCOPE_MATCH_THRESHOLD", "0.40")

# Rewrite lexical continuity, calibrated 2026-08-20 against the labelled
# pairs in eval/rewrite_cases.json with character (2,4) Jaccard overlap.
# Valid normalizations score 0.0756 (a mixed-language Work From Home
# rewrite) to 0.3178, while a rewrite that replaces the question wholesale
# scores 0.0086 or less. 0.05 sits inside that gap. The number is
# deliberately a floor against wholesale replacement, not the main defence:
# anchor, topic, and injection rules reject drift that stays lexically
# close, such as 500 baht becoming 5,000, and one labelled drift case
# scores 0.0570 -- above this floor and stopped by the topic rule alone.
# Held-out data was not consulted.
REWRITE_CONTINUITY_THRESHOLD: float = _env_float(
    "REWRITE_CONTINUITY_THRESHOLD", "0.05"
)

TOP_K: int = _env_int("TOP_K", "3")
MAX_QUERY_CHARS: int = _env_int("MAX_QUERY_CHARS", "500")

# A relative CORPUS_DIR is anchored to the project root so the loader works
# from any working directory; an absolute value replaces the anchor entirely
# per pathlib joining rules.
CORPUS_DIR: str = str(_PROJECT_ROOT / _env_str("CORPUS_DIR", "data/docs"))

# Destination of the append-only fallback/blocked telemetry, anchored like
# CORPUS_DIR. Tests inject a temporary path instead of overriding this.
FALLBACK_LOG_PATH: str = str(
    _PROJECT_ROOT / _env_str("FALLBACK_LOG_PATH", "logs/fallback_queries.jsonl")
)

# Demo-only switch for the persistent audit surface. The JSONL sink holds
# raw employee questions across every session, so the default keeps it out
# of the running app entirely; an operator turns it on deliberately for a
# local walkthrough. This is a feature flag, not authorization: it is not
# authentication, not RBAC, and grants nothing per-user (Finding 7).
ENABLE_OPS_VIEW: bool = _env_flag("ENABLE_OPS_VIEW", "false")


def has_llm_credential() -> bool:
    """Report whether an LLM credential is configured, without reading it.

    Secrets reach the rest of the process only through this module
    (AGENTS.md section 7), and the provider SDK reads the variable itself,
    so nothing here needs the value -- only its presence. The check runs
    per call rather than at import so a credential added to the
    environment after start-up is picked up, and so tests can exercise
    both states.

    Returns:
        True when ``OPENAI_API_KEY`` is set to a non-blank value.
    """
    return bool((os.getenv("OPENAI_API_KEY") or "").strip())

def _require_unit_interval(name: str, value: float) -> None:
    """Reject a threshold outside the valid cosine-similarity range.

    Cosine similarity over non-negative TF-IDF vectors lives in [0, 1], so
    any threshold outside that interval can never match a real score.

    Args:
        name: Environment variable name used in the error message.
        value: Parsed threshold value.

    Raises:
        ValueError: If the value falls outside [0.0, 1.0].
    """
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}")


_require_unit_interval("REWRITE_FLOOR", REWRITE_FLOOR)
_require_unit_interval("DIRECT_ANSWER_THRESHOLD", DIRECT_ANSWER_THRESHOLD)
_require_unit_interval("FINAL_ANSWER_THRESHOLD", FINAL_ANSWER_THRESHOLD)
_require_unit_interval("SCOPE_MATCH_THRESHOLD", SCOPE_MATCH_THRESHOLD)
_require_unit_interval(
    "REWRITE_CONTINUITY_THRESHOLD", REWRITE_CONTINUITY_THRESHOLD
)
if REWRITE_FLOOR > DIRECT_ANSWER_THRESHOLD:
    raise ValueError(
        "REWRITE_FLOOR must not exceed DIRECT_ANSWER_THRESHOLD; the medium "
        "band would otherwise be empty and no query could ever be rewritten"
    )
if FINAL_ANSWER_THRESHOLD < DIRECT_ANSWER_THRESHOLD:
    raise ValueError(
        "FINAL_ANSWER_THRESHOLD must not sit below DIRECT_ANSWER_THRESHOLD; "
        "the expanded score is max-pooled over the original query plus its "
        "rewrites, so a lower final gate would let the rewrite branch answer "
        "queries the direct branch already refused on the same score"
    )
if TOP_K <= 0:
    raise ValueError("TOP_K must be greater than zero")
if MAX_QUERY_CHARS <= 0:
    raise ValueError("MAX_QUERY_CHARS must be greater than zero")
if LLM_TIMEOUT_SECONDS <= 0:
    raise ValueError("LLM_TIMEOUT_SECONDS must be greater than zero")
if LLM_MAX_RETRIES < 0:
    raise ValueError("LLM_MAX_RETRIES must not be negative")
if LLM_REWRITE_TIMEOUT_SECONDS <= 0:
    raise ValueError("LLM_REWRITE_TIMEOUT_SECONDS must be greater than zero")
if LLM_REWRITE_MAX_RETRIES < 0:
    raise ValueError("LLM_REWRITE_MAX_RETRIES must not be negative")
