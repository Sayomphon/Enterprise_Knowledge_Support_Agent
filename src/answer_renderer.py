"""Deterministic rendering of a validated candidate answer.

The reporter returns claims and source ids; the employee reads sentences
with ``[SOURCE-ID]`` markers. That last step is done here, from validated
ids only, so the markup in the public answer is machine-generated and
identical for identical input. Keeping it in ``src`` rather than in the
CLI and the Streamlit app means both surfaces render the same text and
neither owns business logic (AGENTS.md section 3).

Call this only after ``src.guardrails.citation_validator`` accepted the
candidate: rendering an unvalidated candidate would put model output in
front of an employee, which is exactly what the candidate/answer split
exists to prevent.
"""

from __future__ import annotations

from src.schemas import GroundedAnswer

CLAIM_SEPARATOR = "\n"


def render_answer(candidate: GroundedAnswer) -> str:
    """Render a validated candidate answer as the public answer text.

    Args:
        candidate: A candidate the validator has already accepted, so
            every claim carries text and at least one valid source id.

    Returns:
        One line per claim, each followed by its citations in sorted id
        order. Sorting is total so the same candidate always renders the
        same string, which keeps the answer diffable in review and in
        tests.
    """
    return CLAIM_SEPARATOR.join(
        f"{claim.text.strip()} {_citation_markup(claim.normalized_source_ids())}"
        for claim in candidate.claims
    )


def _citation_markup(source_ids: tuple[str, ...]) -> str:
    """Render one claim's validated ids as ``[ID]`` markers."""
    return " ".join(f"[{source_id}]" for source_id in sorted(source_ids))
