"""Immutable domain objects and the LangGraph pipeline state contract.

This module sits at the bottom of the dependency graph and imports nothing
from the rest of ``src`` (AGENTS.md section 3). Domain objects are frozen
dataclasses so every state transition in the pipeline is auditable; the
pipeline state is a ``TypedDict`` whose optional fields mirror the routes
that actually populate them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from typing_extensions import NotRequired, TypedDict

SourceType = Literal["policy", "chat"]

# Terminal or intermediate branch of the pipeline graph (AGENTS.md section 16).
Route = Literal[
    "blocked",
    "direct_answer",
    "rewrite",
    "fallback",
    "answered",
]


@dataclass(frozen=True)
class Document:
    """One corpus document loaded from Markdown with YAML frontmatter.

    Attributes:
        source_id: Stable citation identifier, for example ``"HR-001"``.
            This exact string appears in ``[SOURCE-ID]`` answer citations.
        title: Human-readable document title shown beside citations.
        source_type: Corpus stratum: ``"policy"`` is authoritative text,
            ``"chat"`` is noisy conversational evidence.
        content: Markdown body with the frontmatter block removed.
    """

    source_id: str
    title: str
    source_type: SourceType
    content: str


@dataclass(frozen=True)
class RetrievedDocument:
    """A corpus document paired with its retrieval score for one query.

    Attributes:
        source_id: Citation identifier of the underlying document.
        title: Title of the underlying document.
        source_type: Stratum of the underlying document.
        content: Markdown body of the underlying document.
        score: Cosine similarity between query and document. This is a
            similarity heuristic, never a probability that the answer is
            correct (AGENTS.md section 4, invariant 6).
    """

    source_id: str
    title: str
    source_type: SourceType
    content: str
    score: float


class PipelineState(TypedDict):
    """Shared state flowing through the LangGraph pipeline.

    Only ``query`` exists on every route. Every other field is
    ``NotRequired`` because each route populates a different subset: a
    blocked request never carries retrieval results, and a fallback never
    carries an answer. Keeping the contract partial matches the runtime
    reality instead of forcing placeholder values into unrelated routes.
    """

    query: str

    # Routing outcome and guardrail verdict.
    route: NotRequired[Route]
    guardrail_reason: NotRequired[str]

    # Retrieval results shared by the answer-bearing routes.
    retrieved: NotRequired[list[RetrievedDocument]]
    raw_retrieval_score: NotRequired[float]

    # Populated only on the medium-band rewrite path.
    rewritten_queries: NotRequired[list[str]]
    rewrite_failed: NotRequired[bool]
    expanded_retrieval_score: NotRequired[float]

    # Populated only when the Reporter produced a validated answer.
    answer: NotRequired[str]
    valid_citations: NotRequired[list[str]]

    # Populated only when the request degraded to the fallback response.
    fallback_reason: NotRequired[str]
