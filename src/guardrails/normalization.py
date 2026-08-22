"""Canonical folding shared by the deterministic contract checks.

The contract checks are written in ASCII -- a figure is a run of digits,
a citation is a bracketed id -- while the attacker picks the spelling.
"lot dai 10 wan" written with Thai numerals, a fullwidth bracket, or a
CJK one renders on the employee's screen exactly like the plain form and
misses every pattern written in ASCII, so a claim can state a figure
nothing checks and carry a source id nothing validated.

This module folds those spellings into one before any pattern runs. It
is deliberately a separate concept from ``text_similarity``: that module
measures how alike two strings are, this one decides what a string IS,
and the guardrail's own hardening folds a query for RULE matching and
never leaves the guardrail. Digits are folded through
``unicodedata.decimal`` rather than a table of scripts, so Thai,
Arabic-Indic, Devanagari and fullwidth digits are covered by the same
line and a script nobody listed cannot slip through.

It imports nothing from the rest of ``src`` (AGENTS.md section 3).
"""

from __future__ import annotations

import unicodedata

# NFKC folds fullwidth and other compatibility spellings into plain
# ASCII, but it also splits Thai SARA AM into nikhahit + sara aa, and NFC
# does not put it back because that character is a composition exclusion.
# Recomposing keeps corpus wording byte-identical through the fold, which
# matters because the span check compares folded claim text against
# folded document text.
_THAI_SARA_AM = "ำ"
_THAI_DECOMPOSED_SARA_AM = "ํา"

# Bracket and dash spellings NFKC leaves alone, grouped by the ASCII
# character they render as. A declarative table rather than a chain of
# replacements (AGENTS.md section 6.1); the fullwidth members are absent
# because NFKC already folds those. Square-bracket lookalikes are what a
# fake citation is built from, and the dash family is what separates the
# letters from the digits inside one.
_PUNCTUATION_FOLDING: dict[str, str] = {
    "[": "【〔〖〘〚⟦〈⟨",
    "]": "】〕〗〙〛⟧〉⟩",
    "-": "‐‑‒–—―−﹘",
}

_PUNCTUATION_TRANSLATION = {
    ord(lookalike): ascii_form
    for ascii_form, lookalikes in _PUNCTUATION_FOLDING.items()
    for lookalike in lookalikes
}


def nfkc_fold(text: str) -> str:
    """Apply compatibility normalization without damaging Thai wording.

    Args:
        text: Any string; raw input, claim text, or a document body.

    Returns:
        The NFKC-normalized text with Thai SARA AM recomposed, so a word
        such as "kham-tham" survives the fold as the single character the
        corpus and the queries are written with.
    """
    return unicodedata.normalize("NFKC", text).replace(
        _THAI_DECOMPOSED_SARA_AM, _THAI_SARA_AM
    )


def fold_for_contract_checks(text: str) -> str:
    """Fold one string into the form every contract check compares.

    Args:
        text: Claim text, rewrite candidate, query, or document body.
            Both sides of every comparison must pass through here, or
            the fold would create the mismatch it exists to remove.

    Returns:
        The text with compatibility spellings, decimal digits of any
        script, and bracket and dash lookalikes rewritten as their ASCII
        form. Case and whitespace are left alone: this fold decides what
        a character IS, and the callers that also need case-insensitive
        or whitespace-insensitive comparison compose it with
        ``text_similarity.normalize_for_matching``.
    """
    folded = nfkc_fold(text).translate(_PUNCTUATION_TRANSLATION)
    return _folded_digits(folded)


def _folded_digits(text: str) -> str:
    """Rewrite every decimal digit as the ASCII digit it denotes.

    Args:
        text: Text already through ``nfkc_fold``.

    Returns:
        The text with each decimal character replaced by its value.
        Non-digits and digit-like characters that carry no decimal value
        are returned untouched, so nothing outside a figure moves.
    """
    folded: list[str] = []
    for character in text:
        # -1 rather than an exception: the default keeps the loop over a
        # whole document body free of try/except per character.
        value = unicodedata.decimal(character, -1)
        folded.append(character if value < 0 else str(value))
    return "".join(folded)
