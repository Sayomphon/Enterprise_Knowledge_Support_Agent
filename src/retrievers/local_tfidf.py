"""Character n-gram TF-IDF retrieval over the in-memory corpus.

Fully local and deterministic: no embedding API, no model download, no
network call, no vector store. The retriever only computes similarity
scores; routing decisions against thresholds belong to the graph layer,
so no threshold appears anywhere in this module.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.schemas import Document, RetrievedDocument


class LocalTfidfRetriever:
    """Doc-level retrieval backed by a character n-gram TF-IDF index."""

    def __init__(
        self,
        documents: Sequence[Document],
        ngram_range: tuple[int, int] = (2, 5),
        title_weight: int = 1,
    ) -> None:
        """Build the TF-IDF index exactly once over the supplied corpus.

        Args:
            documents: Validated corpus documents from the loader. Only
                the active ones are indexed. The index is fitted here at
                construction time; ``search`` only transforms queries and
                never rebuilds it.
            ngram_range: Character n-gram span. Injectable so the
                calibration ablation can sweep candidate configurations
                without editing this module. See ``src/config.py`` for the
                sweep that chose the default.
            title_weight: How many times a document's title is repeated
                in its searchable string. Titles carry the most
                discriminative policy keywords in this corpus, so
                repeating one raises the score of a short query that
                names the topic without touching a threshold.

                The default stays 1: the 2026-08-22 ablation over
                {(2,4), (2,5), (3,5)} x {1, 2, 3} on the 25-case
                calibration set found that weighting titles does lift
                Coverage to 16/16, but that it also reorders an exact
                policy question -- "how do I claim a taxi fare after
                OT" ranks the chat transcript above FIN-001, because a
                chat title is short and echoes the question. Ranking a
                transcript over the policy it illustrates is a worse
                outcome than the case it recovers. The (2,4) column
                scored the same and pushed the strongest generic
                out-of-domain query to 0.1029, above ``REWRITE_FLOOR``,
                which the score itself should separate. The parameter is
                kept because it is how the ablation was run and how the
                next one will be; see eval/BASELINE.md for the table.

        Raises:
            ValueError: If no active document remains to index, or
                ``title_weight`` is below one.
        """
        if not documents:
            raise ValueError(
                "LocalTfidfRetriever requires at least one document"
            )
        # Retired documents are excluded from the index, not merely from
        # the answer. Leaving them in lets a superseded policy take a
        # top-k slot from the active one that replaced it, and -- because
        # the band router reads the top-1 score -- lets it decide the
        # route on the strength of text that may never support an answer.
        # The full corpus, retired entries included, still reaches
        # evidence selection through ``documents_by_id`` for canonical
        # link resolution, which rejects them there on its own terms.
        self._documents = tuple(
            document for document in documents if document.status == "active"
        )
        if not self._documents:
            raise ValueError(
                "LocalTfidfRetriever requires at least one active document"
            )
        if title_weight < 1:
            raise ValueError("title_weight must be at least 1")
        # Public read-only telemetry: observability surfaces report the
        # active index configuration from here instead of guessing.
        self.ngram_range = ngram_range
        self.title_weight = title_weight
        # Character n-grams instead of word tokenisation: Thai has no
        # reliable whitespace word boundaries, so a word analyzer would
        # need a segmentation model and still break on informal spelling.
        # Overlapping character fragments keep a transposed-tone-mark
        # misspelling of "annual leave" close to its correct spelling,
        # because only the fragments covering the swapped characters stop
        # matching.
        self._vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            sublinear_tf=True,
            lowercase=True,
        )
        # Title and body share one searchable string because titles carry
        # the most discriminative policy keywords in this small corpus.
        # Repeating the title is how weighting is expressed here: the
        # vectorizer is fitted on text, so a field weight has to be a
        # property of that text rather than of a separate matrix.
        self._matrix = self._vectorizer.fit_transform(
            [
                "\n".join(
                    [document.title] * title_weight + [document.content]
                )
                for document in self._documents
            ]
        )

    def search(
        self, queries: Sequence[str], top_k: int
    ) -> list[RetrievedDocument]:
        """Return the top-k documents by max-pooled cosine similarity.

        One code path serves both pipeline retrieval stages: original
        retrieval passes a single query, expanded retrieval passes the
        original plus every rewrite, and each document keeps the maximum
        similarity it reached across all supplied queries.

        Args:
            queries: One or more query strings to score against the index.
                The first entry is treated as the employee's own query and
                the rest as generated rewrites, which is the order the
                graph always uses.
            top_k: Number of documents to return, capped at corpus size.

        Returns:
            Retrieved documents ordered by score descending, tie-broken by
            ascending ``source_id`` so the ranking is total and any rerun
            reproduces it exactly. Each result records which query earned
            its score, so a reviewer can see when a document ranked only
            because of a rewrite.

        Raises:
            ValueError: If ``queries`` is empty or ``top_k`` is below one.
        """
        if not queries:
            raise ValueError("search requires at least one query")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        # NFC mirrors the loader's corpus normalization so visually equal
        # Thai strings can never miss on byte-level encoding differences.
        normalized_queries = [
            unicodedata.normalize("NFC", query) for query in queries
        ]
        query_vectors = self._vectorizer.transform(normalized_queries)
        scores = cosine_similarity(query_vectors, self._matrix)
        pooled_scores = scores.max(axis=0)
        best_query_indices = scores.argmax(axis=0)
        ranked = sorted(
            zip(self._documents, pooled_scores, best_query_indices),
            key=lambda triple: (-triple[1], triple[0].source_id),
        )
        return [
            RetrievedDocument(
                source_id=document.source_id,
                title=document.title,
                source_type=document.source_type,
                content=document.content,
                score=float(score),
                authority=document.authority,
                status=document.status,
                topics=document.topics,
                canonical_source_ids=document.canonical_source_ids,
                matched_query=queries[int(query_index)],
                matched_query_type=(
                    "original" if int(query_index) == 0 else "rewrite"
                ),
            )
            for document, score, query_index in ranked[:top_k]
        ]
