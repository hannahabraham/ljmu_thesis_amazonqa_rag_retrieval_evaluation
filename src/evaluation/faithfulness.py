"""Lightweight faithfulness / groundedness / hallucination metrics.

These are *lexical* approximations computed from the retrieved context and the
generated answer. They are cheap (no LLM calls), so we can compute them on every
(pipeline, k) cell — unlike RAGAS faithfulness which only runs at k=5.

Conventions
-----------
* `groundedness`     — fraction of answer content tokens that appear somewhere in
                       the retrieved context. 1.0 = every meaningful word is
                       supported lexically; 0.0 = the answer is unsupported.
* `hallucination_rate` per row — `1 - groundedness`. Aggregated as the mean
                       across rows (refusals are excluded by default since
                       they're not making factual claims).

These metrics are deliberately conservative: lexical overlap underestimates
groundedness for paraphrases, so the *trend* across pipelines is more reliable
than the absolute score. RAGAS faithfulness (LLM-as-judge) remains the headline
faithfulness number.
"""
from __future__ import annotations

import re
import string
from typing import Iterable, Sequence

# Common English stop words — small list, kept inline to avoid an NLTK download.
_STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are as at be because been
    before being below between both but by can did do does doing down during each
    few for from further had has have having he her here hers herself him himself
    his how i if in into is it its itself just me more most my myself no nor not
    now of off on once only or other our ours ourselves out over own same she
    should so some such than that the their theirs them themselves then there
    these they this those through to too under until up very was we were what when
    where which while who whom why will with you your yours yourself yourselves
    """.split()
)

_TOKEN_RX = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    if not text:
        return []
    cleaned = text.lower().translate(str.maketrans("", "", string.punctuation))
    return _TOKEN_RX.findall(cleaned)


def _content_tokens(text: str) -> list[str]:
    return [t for t in _tokens(text) if t not in _STOPWORDS and len(t) > 1]


def groundedness(answer: str, contexts: Sequence[str]) -> float:
    """Fraction of answer content tokens supported by the retrieved context.

    Returns NaN for empty answers (refusals) so they don't bias the mean.
    """
    answer_tokens = _content_tokens(answer)
    if not answer_tokens:
        return float("nan")
    context_tokens: set[str] = set()
    for ctx in contexts:
        context_tokens.update(_content_tokens(ctx))
    if not context_tokens:
        return 0.0
    supported = sum(1 for t in answer_tokens if t in context_tokens)
    return supported / len(answer_tokens)


def hallucination_rate_row(answer: str, contexts: Sequence[str]) -> float:
    """Per-row hallucination = 1 - groundedness. NaN for empty answers."""
    g = groundedness(answer, contexts)
    if g != g:  # NaN
        return float("nan")
    return 1.0 - g


def aggregate_hallucination_rate(
    answers: Iterable[str],
    context_lists: Iterable[Sequence[str]],
    refused_flags: Iterable[bool] | None = None,
) -> float:
    """Mean hallucination rate across rows.

    Refusals are excluded by default — a refusal asserts nothing, so it isn't a
    hallucination. Pass `refused_flags=None` to include every row.
    """
    answers_list = list(answers)
    context_lists_list = list(context_lists)
    refused_list = list(refused_flags) if refused_flags is not None else [False] * len(answers_list)
    if not (len(answers_list) == len(context_lists_list) == len(refused_list)):
        raise ValueError("answers, contexts, refused_flags must align in length")

    rates: list[float] = []
    for ans, ctx, refused in zip(answers_list, context_lists_list, refused_list):
        if refused:
            continue
        rate = hallucination_rate_row(ans, ctx)
        if rate == rate:  # not NaN
            rates.append(rate)
    if not rates:
        return float("nan")
    return sum(rates) / len(rates)


def aggregate_groundedness(
    answers: Iterable[str],
    context_lists: Iterable[Sequence[str]],
    refused_flags: Iterable[bool] | None = None,
) -> float:
    """Mean groundedness across non-refusal rows."""
    answers_list = list(answers)
    context_lists_list = list(context_lists)
    refused_list = list(refused_flags) if refused_flags is not None else [False] * len(answers_list)

    scores: list[float] = []
    for ans, ctx, refused in zip(answers_list, context_lists_list, refused_list):
        if refused:
            continue
        g = groundedness(ans, ctx)
        if g == g:
            scores.append(g)
    if not scores:
        return float("nan")
    return sum(scores) / len(scores)
