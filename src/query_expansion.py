"""Deterministic alias expansion for medium-band queries.

The medium band is the only place where routing depends on a model: a
score between the rewrite floor and the direct threshold buys one LLM
rewrite, and whether the request is answered then depends on what the
provider returned that minute. That is the one non-reproducible edge of
an otherwise deterministic pipeline, and it costs a call on every slang
or typo question.

Most of those questions do not need a model. The scope gate has already
resolved the query to supported topics, and each topic carries the
corpus's own vocabulary in ``SUPPORTED_TOPIC_ALIASES`` -- the formal
spelling of what the employee wrote informally. Searching the original
query *plus* those aliases lets the retriever's max-pool pick whichever
phrasing matches the corpus better.

Expansion, never substitution. Rewriting "bai-set" into "bai-set-rap-ngoen"
inside the query would create a new failure mode (a wrong replacement
destroys recall) and a mapping to maintain apart from the catalog.
Adding a variant cannot do that: the original query is always searched
too, and the retriever keeps the maximum score per document, so an
unhelpful variant is worth exactly zero rather than negative.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from src.guardrails.scope_validator import SUPPORTED_TOPIC_ALIASES
from src.guardrails.text_similarity import containment
from src.schemas import RetrievedDocument

# Aliases kept per topic. The catalog lists up to twelve phrasings of one
# topic, and pooling all of them into a variant dilutes the fragments that
# actually distinguish the documents: the point of the variant is to add
# the corpus's wording for THIS question, not to restate the whole topic.
_ALIASES_PER_TOPIC = 3


def alias_expansion_variants(
    query: str, topics: Sequence[str]
) -> list[str]:
    """Build the deterministic search variants for one scoped query.

    Args:
        query: The guardrail-normalized user query.
        topics: Supported topics the scope gate resolved, which is why
            this cannot run before that gate: aliases of a topic the
            employee never asked about would pull unrelated documents up
            the ranking.

    Returns:
        Up to two variants per topic -- the query with that topic's
        best-matching aliases appended, and those aliases alone -- with
        duplicates removed and the ordering fixed, so the same query and
        topics always produce the same searches. Empty when no topic
        resolved, which is also the low band's own answer.
    """
    variants: list[str] = []
    for topic in topics:
        aliases = _best_aliases(query, topic)
        if not aliases:
            continue
        joined = " ".join(aliases)
        # Two shapes, because they fail differently. The combined variant
        # keeps the employee's own terms beside the corpus wording and
        # wins when the query is already close; the alias-only variant
        # drops the noise of a chatty question ("...gee-wan-a") entirely
        # and wins when that noise is most of the string.
        variants.append(f"{query} {joined}")
        variants.append(joined)
    return _deduplicated(variants)


def label_alias_matches(
    documents: Sequence[RetrievedDocument],
    alias_queries: Sequence[str],
    original_query: str,
) -> list[RetrievedDocument]:
    """Mark the documents whose score came from an alias variant.

    Provenance is the reason expansion is safe to run without the rewrite
    validator: an alias variant is configuration data this repository
    owns, but a reviewer still has to see when a document ranked on
    catalog wording rather than on what the employee typed. The retriever
    reports which query earned each score, so the label is read off that
    rather than guessed.

    Args:
        documents: Results of an expanded search, in ranking order.
        alias_queries: The alias variants included in that search.
        original_query: The employee's own query, which stays "original"
            even in the impossible case of a variant equal to it.

    Returns:
        The same documents, with ``matched_query_type`` set to ``"alias"``
        wherever the winning query was a variant.
    """
    variants = set(alias_queries) - {original_query}
    return [
        replace(document, matched_query_type="alias")
        if document.matched_query in variants
        else document
        for document in documents
    ]


def _best_aliases(query: str, topic: str) -> tuple[str, ...]:
    """Pick the aliases of one topic that best match this query.

    Args:
        query: The guardrail-normalized user query.
        topic: A topic key; an unknown key yields no aliases rather than
            an error, because the caller reads topics out of pipeline
            state rather than from a literal.

    Returns:
        Up to ``_ALIASES_PER_TOPIC`` aliases, best match first. Ties break
        on the alias text so the ordering is total and reproducible.
    """
    aliases = SUPPORTED_TOPIC_ALIASES.get(topic, ())
    ranked = sorted(
        aliases, key=lambda alias: (-containment(alias, query), alias)
    )
    return tuple(ranked[:_ALIASES_PER_TOPIC])


def _deduplicated(variants: Sequence[str]) -> list[str]:
    """Drop repeated variants while keeping the first occurrence's place.

    Two topics can share an alias, and a query that already contains its
    aliases verbatim can produce the combined variant twice. A duplicate
    query costs a column in the similarity matrix and buys nothing: the
    score is max-pooled.
    """
    seen: set[str] = set()
    unique: list[str] = []
    for variant in variants:
        if variant not in seen:
            seen.add(variant)
            unique.append(variant)
    return unique
