"""Retrieval metrics against the golden evidence_doc_id.

Single-evidence regime: each gold row carries one canonical evidence_doc_id, so
Hit@K == Recall@K (both are 0/1) and nDCG@K reduces to 1/log2(rank+1) when the
evidence sits in the top-k, else 0. Implementations are written in the general
multi-relevant form so the metrics still behave correctly if the schema later
gains multiple relevant docs per question.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def _as_list(retrieved_doc_ids: Sequence[str]) -> list[str]:
    return list(retrieved_doc_ids)


def _gold_set(evidence: object) -> set[str]:
    """Accept either a single doc_id (str) or an iterable of doc_ids."""
    if evidence is None:
        return set()
    if isinstance(evidence, str):
        return {evidence}
    if isinstance(evidence, Iterable):
        return {str(x) for x in evidence if x is not None}
    return {str(evidence)}


def hit_at_k(retrieved_doc_ids: Sequence[str], evidence_doc_id: object, k: int) -> int:
    gold = _gold_set(evidence_doc_id)
    return int(any(d in gold for d in _as_list(retrieved_doc_ids)[:k]))


def recall_at_k(retrieved_doc_ids: Sequence[str], evidence_doc_id: object, k: int) -> float:
    """Fraction of relevant docs retrieved in top-k. Binary in single-evidence mode.

    Sentence-level retrievers return multiple chunks per parent doc; we count each
    parent doc once so recall stays in [0, 1].
    """
    gold = _gold_set(evidence_doc_id)
    if not gold:
        return float("nan")
    found = len(set(_as_list(retrieved_doc_ids)[:k]) & gold)
    return found / len(gold)


def reciprocal_rank(retrieved_doc_ids: Sequence[str], evidence_doc_id: object) -> float:
    gold = _gold_set(evidence_doc_id)
    if not gold:
        return float("nan")
    for idx, doc_id in enumerate(_as_list(retrieved_doc_ids), start=1):
        if doc_id in gold:
            return 1.0 / idx
    return 0.0


def dcg_at_k(retrieved_doc_ids: Sequence[str], evidence_doc_id: object, k: int) -> float:
    gold = _gold_set(evidence_doc_id)
    if not gold:
        return float("nan")
    total = 0.0
    seen: set[str] = set()
    for rank, doc_id in enumerate(_as_list(retrieved_doc_ids)[:k], start=1):
        if doc_id in gold and doc_id not in seen:
            total += 1.0 / math.log2(rank + 1)
            seen.add(doc_id)
    return total


def ndcg_at_k(retrieved_doc_ids: Sequence[str], evidence_doc_id: object, k: int) -> float:
    """nDCG@k assuming binary relevance (rel=1 for gold, 0 otherwise)."""
    gold = _gold_set(evidence_doc_id)
    if not gold:
        return float("nan")
    dcg = dcg_at_k(retrieved_doc_ids, evidence_doc_id, k)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg
