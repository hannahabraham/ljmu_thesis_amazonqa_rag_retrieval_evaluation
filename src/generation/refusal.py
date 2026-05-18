"""Refusal / abstention detection for RAG answers.

Strategy
--------
1. Empty / whitespace-only answers count as refusals.
2. Answers that start with "Yes" or "No" are factual, even if they later
   disclaim partial information ("Yes, the reviews don't say exactly how long").
3. Refusal patterns must match the *leading clause* (everything up to the
   first sentence break or contrast conjunction such as "but"/"however"/
   "although"/"though"/"while"). This prevents factual answers that mention
   a gap mid-sentence from being mis-classified.
"""
from __future__ import annotations

import re

_PATTERNS: tuple[str, ...] = (
    r"\bnot enough information\b",
    r"\binsufficient (information|context|evidence|detail)\b",
    r"\b(cannot|can'?t|unable to) (determine|tell|answer|find|verify|conclude)\b",
    r"\b(reviews?|context|evidence) (do(?:es)? not|don'?t|fail to) "
    r"(provide|contain|mention|say|specify|indicate|address|state)\b",
    r"\bno (information|mention|evidence|details?) "
    r"(?:available |provided )?"
    r"(?:in|from|within|among)? ?(?:the )?(?:reviews?|context|evidence)\b",
    r"\bunclear (from|in|based on) the (reviews?|context|evidence)\b",
    r"\bnot (specified|mentioned|stated|addressed|covered) (in|by) the (reviews?|context)\b",
    r"\bthere is no (information|evidence|mention) (in|from|within) the (reviews?|context)\b",
)

_COMPILED = tuple(re.compile(pattern, re.IGNORECASE) for pattern in _PATTERNS)

# Split on sentence breaks (. ; ? !) or contrast conjunctions. We only examine
# the leading clause so factual answers like "Yes, but the reviews don't say X"
# are not mis-classified.
_LEADING_SPLIT = re.compile(
    r"(?:[.;?!]|\b(?:but|however|although|though|while|whereas)\b)",
    re.IGNORECASE,
)

# Yes/No as a *stance* (followed by punctuation or end-of-string), distinguished
# from "No"/"Yes" used as determiners ("No information in the reviews").
_STARTS_WITH_YESNO = re.compile(r"^\s*(yes|no)\b\s*([,.;:!?]|$)", re.IGNORECASE)


def _leading_clause(text: str) -> str:
    """Return the answer's leading clause (up to the first break or contrast)."""
    parts = _LEADING_SPLIT.split(text, maxsplit=1)
    return parts[0] if parts else text


def is_refusal(answer: str) -> bool:
    """Return True if the answer signals abstention/refusal to answer."""
    if not answer or not answer.strip():
        return True
    text = answer.strip()
    if _STARTS_WITH_YESNO.match(text):
        return False
    leading = _leading_clause(text)
    return any(rx.search(leading) for rx in _COMPILED)
