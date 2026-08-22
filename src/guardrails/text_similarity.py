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

import re
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


def required_tokens(phrase: str) -> tuple[str, ...]:
    """Return the words a whitespace-delimited phrase must find verbatim.

    Character n-grams exist because Thai has no reliable word boundaries,
    and on a language that HAS them the measure over-fires: the fragments
    of "sick leave" that appear in "how many annual leave days" all come
    from the single word "leave", yet they are half the phrase, which
    scored the sick-leave topic 0.5 on an annual-leave question. Where a
    token boundary really exists, it is evidence and this uses it.

    Args:
        phrase: A normalized reference phrase, for example a topic alias.

    Returns:
        The phrase's whitespace tokens when it is written in ASCII and
        holds more than one of them, else an empty tuple. Thai and mixed
        aliases return nothing and keep the pure n-gram behaviour, which
        is what the scope threshold was calibrated on.
    """
    tokens = phrase.split()
    return tuple(tokens) if len(tokens) > 1 and phrase.isascii() else ()


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
        [0.0, 1.0]. Returns 1.0 when the phrase occurs verbatim, 0.0 when
        the phrase is too short to produce any n-gram, and 0.0 when a
        multi-word ASCII phrase is missing one of its own words.
    """
    normalized_phrase = normalize_for_matching(phrase)
    normalized_text = normalize_for_matching(text)
    return containment_of_ngrams(
        normalized_phrase,
        character_ngrams(normalized_phrase, ngram_range),
        normalized_text,
        character_ngrams(normalized_text, ngram_range),
        tokens=required_tokens(normalized_phrase),
    )


def containment_of_ngrams(
    phrase: str,
    phrase_ngrams: set[str],
    text: str,
    text_ngrams: set[str],
    tokens: tuple[str, ...] = (),
) -> float:
    """Score containment from fragments the caller already computed.

    ``containment`` re-normalizes and re-fragments both sides on every
    call, which is wasted work for a catalog of constant aliases scored
    against one query: the scope gate was rebuilding 61 constant alias
    sets, and the query's own set 61 times, for every question.

    Args:
        phrase: Normalized reference phrase, for the verbatim check.
        phrase_ngrams: Fragments of ``phrase``.
        text: Normalized text to search, for the verbatim check.
        text_ngrams: Fragments of ``text``.
        tokens: Words the phrase must contribute whole, from
            ``required_tokens``. Empty means no token gate, which is the
            behaviour every Thai alias keeps.

    Returns:
        The same value ``containment`` would return for the two strings.
    """
    if tokens and not _all_tokens_present(text, tokens):
        return 0.0
    if phrase and phrase in text:
        return 1.0
    if not phrase_ngrams:
        return 0.0
    return len(phrase_ngrams & text_ngrams) / len(phrase_ngrams)


def _all_tokens_present(text: str, tokens: tuple[str, ...]) -> bool:
    """Report whether every token occurs in the text as a whole word.

    Word boundaries rather than plain substrings: "leave" must not be
    satisfied by "leaves" belonging to another phrase, and a token glued
    inside a longer run of letters is not the word the alias named.
    """
    return all(
        re.search(rf"\b{re.escape(token)}\b", text) is not None
        for token in tokens
    )


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
