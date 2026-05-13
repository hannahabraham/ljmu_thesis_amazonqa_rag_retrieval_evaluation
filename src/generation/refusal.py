"""Refusal / abstention detection for RAG answers.

Hand-validated on 50 labelled samples (see scripts/validate_refusal_detector.py).
Target: precision >0.95, recall >0.95. Re-validate if the prompt template changes.
"""
from __future__ import annotations

import re

_PATTERNS: tuple[str, ...] = (
    r"\bnot enough information\b",
    r"\binsufficient (information|context|evidence|detail)\b",
    r"\b(cannot|can'?t|unable to) (determine|tell|answer|find|verify|conclude)\b",
    r"\b(reviews?|context|evidence) (do(?:es)? not|don'?t|fail to) "
    r"(provide|contain|mention|say|specify|indicate|address)\b",
    r"\bno (information|mention|evidence|details?) "
    r"(?:available |provided )?"
    r"(?:in|from|within|among)? ?(?:the )?(?:reviews?|context|evidence)\b",
    r"\bunclear (from|in|based on) the (reviews?|context|evidence)\b",
    r"\b(the )?(available )?reviews? (do(?:es)? not|don'?t|fail to)\b",
    r"\bnot (specified|mentioned|stated|addressed|covered) (in|by) the (reviews?|context)\b",
    r"\bthere is no (information|evidence|mention) (in|from|within) the (reviews?|context)\b",
)

_COMPILED = tuple(re.compile(pattern, re.IGNORECASE) for pattern in _PATTERNS)


def is_refusal(answer: str) -> bool:
    """Return True if the answer signals abstention/refusal to answer."""
    if not answer or not answer.strip():
        return True
    return any(rx.search(answer) for rx in _COMPILED)
