"""Shared retrieval contract for every backend.

Extending retrieval means adding a new class that satisfies this protocol
(AGENTS.md section 6.5), never adding backend branches to the graph or the
router. Keeping the contract in its own module lets a future BM25, hybrid,
or vector backend import it without depending on the TF-IDF implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from src.schemas import RetrievedDocument


@runtime_checkable
class Retriever(Protocol):
    """Contract every retrieval backend must satisfy."""

    def search(
        self, queries: Sequence[str], top_k: int
    ) -> list[RetrievedDocument]:
        """Return the top-k documents, sorted by descending score."""
        ...
