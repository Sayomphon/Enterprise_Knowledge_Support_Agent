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


MODEL_NAME: str = _env_str("MODEL_NAME", "gpt-5-mini")

# PLACEHOLDER thresholds — deliberately NOT calibrated. They exist only so
# the three-band router can be wired and unit-tested before the calibration
# sweep runs. Each value will be replaced by the sweep over
# eval/retrieval_calibration.json, together with a provenance comment
# recording how it was derived (AGENTS.md sections 4.5 and 10).
REWRITE_FLOOR: float = _env_float("REWRITE_FLOOR", "0.10")
DIRECT_ANSWER_THRESHOLD: float = _env_float("DIRECT_ANSWER_THRESHOLD", "0.30")
FINAL_ANSWER_THRESHOLD: float = _env_float("FINAL_ANSWER_THRESHOLD", "0.30")

TOP_K: int = _env_int("TOP_K", "3")
MAX_QUERY_CHARS: int = _env_int("MAX_QUERY_CHARS", "500")

# A relative CORPUS_DIR is anchored to the project root so the loader works
# from any working directory; an absolute value replaces the anchor entirely
# per pathlib joining rules.
CORPUS_DIR: str = str(_PROJECT_ROOT / _env_str("CORPUS_DIR", "data/docs"))

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
if REWRITE_FLOOR > DIRECT_ANSWER_THRESHOLD:
    raise ValueError(
        "REWRITE_FLOOR must not exceed DIRECT_ANSWER_THRESHOLD; the medium "
        "band would otherwise be empty and no query could ever be rewritten"
    )
if TOP_K <= 0:
    raise ValueError("TOP_K must be greater than zero")
if MAX_QUERY_CHARS <= 0:
    raise ValueError("MAX_QUERY_CHARS must be greater than zero")
