"""Local character n-gram similarity shared by the deterministic guards.

Thai has no reliable whitespace word boundaries, so the guards compare
overlapping character fragments instead of tokens -- the same reasoning
that made the retriever a character TF-IDF index. Keeping the two
measures in one module means the scope gate and the rewrite validator
cannot drift apart, and neither needs a model, a network call, or a
trained tokenizer.

Two measures are offered because the guards ask different questions:
containment asks "does this short phrase appear inside that text", while
overlap asks "how much do these two texts share overall".
"""

from __future__ import annotations

import unicodedata

# Fragments of two to four characters. Two-character fragments alone let
# a common Thai word such as "ngoen" (money) inside an unrelated phrase
# carry an alias most of the way to a match; requiring four-character
# fragments as well removes that noise while still tolerating a typo,
# which only breaks the fragments that overlap the misspelled character.
DEFAULT_NGRAM_RANGE = (2, 4)


def normalize_for_matching(text: str) -> str:
    """Fold one string into the form both similarity measures compare.

    Args:
        text: Raw query, alias, or rewrite candidate.

    Returns:
        The NFC-normalized, case-folded text with runs of whitespace
        collapsed to a single space. NFC rather than NFKC keeps Thai
        combining characters in their canonical composed form without
        rewriting compatibility characters the corpus may rely on.
    """
    normalized = unicodedata.normalize("NFC", text).casefold()
    return " ".join(normalized.split())


def character_ngrams(
    text: str, ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE
) -> set[str]:
    """Extract the set of character n-grams of one normalized string.

    Args:
        text: Text to fragment; normalize it first for stable results.
        ngram_range: Inclusive minimum and maximum n-gram length.

    Returns:
        Every distinct n-gram in the requested length range. Empty when
        the text is shorter than the minimum length.
    """
    minimum, maximum = ngram_range
    fragments: set[str] = set()
    for size in range(minimum, maximum + 1):
        for start in range(len(text) - size + 1):
            fragments.add(text[start : start + size])
    return fragments


def containment(
    phrase: str,
    text: str,
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
) -> float:
    """Measure how much of a short phrase appears inside a longer text.

    Containment is asymmetric on purpose: an alias must be found inside
    the query, and a long query must not be penalised for the words it
    adds around the alias.

    Args:
        phrase: The short reference phrase, for example a topic alias.
        text: The text to search, for example a user query.
        ngram_range: N-gram length range for both sides.

    Returns:
        The share of the phrase's n-grams present in the text, in
        [0.0, 1.0]. Returns 1.0 when the phrase occurs verbatim, and 0.0
        when the phrase is too short to produce any n-gram.
    """
    normalized_phrase = normalize_for_matching(phrase)
    normalized_text = normalize_for_matching(text)
    if normalized_phrase and normalized_phrase in normalized_text:
        return 1.0
    phrase_ngrams = character_ngrams(normalized_phrase, ngram_range)
    if not phrase_ngrams:
        return 0.0
    text_ngrams = character_ngrams(normalized_text, ngram_range)
    return len(phrase_ngrams & text_ngrams) / len(phrase_ngrams)


def overlap(
    first: str,
    second: str,
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
) -> float:
    """Measure how much two texts of comparable length share.

    Args:
        first: One text, for example the original query.
        second: The other text, for example a rewrite candidate.
        ngram_range: N-gram length range for both sides.

    Returns:
        The Jaccard similarity of the two n-gram sets, in [0.0, 1.0].
        Returns 0.0 when either side produces no n-gram, so an empty
        candidate can never look similar to anything.
    """
    first_ngrams = character_ngrams(
        normalize_for_matching(first), ngram_range
    )
    second_ngrams = character_ngrams(
        normalize_for_matching(second), ngram_range
    )
    if not first_ngrams or not second_ngrams:
        return 0.0
    union = first_ngrams | second_ngrams
    return len(first_ngrams & second_ngrams) / len(union)
