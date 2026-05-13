"""Generation metrics: SQuAD EM, token F1, ROUGE-L, BERTScore F1, semantic similarity.

Also exposes the v5 ``is_correct`` definition used to populate Table 1's
"Correct Answers" column:

    Correct ≡ (is_answerable AND token_f1 >= F1_THRESHOLD) OR
              (NOT is_answerable AND refused)

Threshold is configurable via CORRECT_F1_THRESHOLD env / settings constant, and
a sensitivity sweep at 0.3 / 0.5 / 0.7 is run by ``scripts/22_final_ranking.py``.
"""
from __future__ import annotations

import math
import re
import string
from collections import Counter
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from config.settings import CORRECT_F1_THRESHOLD


def _safe_text(text: object) -> str:
    """Coerce pandas NaN / None / non-strings to empty string for metric inputs."""
    if text is None:
        return ""
    if isinstance(text, float) and math.isnan(text):
        return ""
    return str(text)


def normalise_answer(text: object) -> str:
    """SQuAD-style normalisation. Tolerates None and NaN (treated as empty)."""
    text = _safe_text(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, gold: str) -> int:
    return int(normalise_answer(prediction) == normalise_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalise_answer(prediction).split()
    gold_tokens = normalise_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pred_tokens)
    r = num_same / len(gold_tokens)
    return 2 * p * r / (p + r)


def rouge_l(prediction: str, gold: str) -> float:
    """ROUGE-L F1 via the rouge_score package; lazy-imported."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return float("nan")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(_safe_text(gold), _safe_text(prediction))["rougeL"].fmeasure


def bertscore_f1(predictions: Iterable[str], golds: Iterable[str]) -> list[float]:
    """BERTScore F1 for a batch; returns NaNs if bert_score isn't installed."""
    preds = [_safe_text(p) for p in predictions]
    refs = [_safe_text(g) for g in golds]
    try:
        from bert_score import score
    except ImportError:
        return [float("nan")] * len(preds)
    if not preds:
        return []
    # bert-score 0.3.13 crashes on empty strings against transformers>=5
    # (RobertaTokenizer.build_inputs_with_special_tokens was removed). It strips
    # input first, so a literal placeholder word is needed (a space gets stripped
    # back to empty). Zero-out those pairs after scoring.
    _PLACEHOLDER = "empty"
    empty = [not p.strip() or not r.strip() for p, r in zip(preds, refs)]
    safe_preds = [p if p.strip() else _PLACEHOLDER for p in preds]
    safe_refs = [r if r.strip() else _PLACEHOLDER for r in refs]
    _, _, f1 = score(safe_preds, safe_refs, lang="en", rescale_with_baseline=False)
    scores = f1.tolist()
    return [0.0 if e else s for s, e in zip(scores, empty)]


@lru_cache(maxsize=1)
def _embedder():
    """Lazy-loaded sentence-transformers encoder for semantic similarity."""
    from sentence_transformers import SentenceTransformer

    from config.settings import EMBEDDING_MODEL

    return SentenceTransformer(EMBEDDING_MODEL)


def semantic_similarity(predictions: Sequence[str], golds: Sequence[str]) -> list[float]:
    """Cosine similarity between prediction and gold-answer embeddings.

    Returns one float in [-1, 1] per pair. Scores are clipped to [0, 1] so they
    align with the rest of the dashboard (and so means are interpretable). NaN if
    the embedder isn't installed.
    """
    preds = [_safe_text(p) for p in predictions]
    refs = [_safe_text(g) for g in golds]
    if not preds:
        return []
    try:
        import numpy as np

        model = _embedder()
        pred_emb = model.encode(preds, convert_to_numpy=True, normalize_embeddings=True)
        ref_emb = model.encode(refs, convert_to_numpy=True, normalize_embeddings=True)
        sims = np.einsum("ij,ij->i", pred_emb, ref_emb)
        return [float(max(0.0, min(1.0, s))) for s in sims]
    except Exception:  # noqa: BLE001 — embedder missing or device error
        return [float("nan")] * len(preds)


def _coerce_bool(value: Any) -> bool:
    """Robustly coerce DataFrame-derived truthy values (0/1, "True", NaN) to bool."""
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "t"}
    return bool(value)


def is_correct(row: Mapping[str, Any], f1_threshold: float = CORRECT_F1_THRESHOLD) -> bool:
    """Table 1 "Correct" definition.

    Answerable rows: model must attempt AND token_f1 must clear the threshold.
    Unanswerable rows: model must refuse.
    """
    answerable = _coerce_bool(row.get("is_answerable"))
    refused = _coerce_bool(row.get("refused"))
    if answerable:
        token_f1 = float(row.get("token_f1") or 0.0)
        return (not refused) and (token_f1 >= f1_threshold)
    return refused


def correct_answers_count(
    per_q: pd.DataFrame, threshold: float = CORRECT_F1_THRESHOLD,
) -> int:
    """Count of rows that satisfy ``is_correct`` at the given threshold."""
    if per_q.empty:
        return 0
    return int(per_q.apply(lambda r: is_correct(r, threshold), axis=1).sum())


def yesno_em(prediction: str, gold: str) -> int:
    """Strict yes/no EM after lowercasing and trimming punctuation."""
    def _first_token(text: str) -> str:
        norm = normalise_answer(text)
        return norm.split(" ")[0] if norm else ""
    return int(_first_token(prediction) == _first_token(gold))
