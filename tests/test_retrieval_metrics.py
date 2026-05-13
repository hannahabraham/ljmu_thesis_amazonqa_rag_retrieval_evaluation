"""Unit tests for retrieval evaluation metrics."""

from __future__ import annotations

from src.evaluation.retrieval_metrics import (
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_hit_at_k() -> None:
    """Test hit@k metric for successful and unsuccessful retrievals."""
    assert hit_at_k(["a", "b", "c"], "b", k=2) == 1
    assert hit_at_k(["a", "b", "c"], "c", k=2) == 0


def test_recall_at_k() -> None:
    """Test recall@k metric behaviour."""
    assert recall_at_k(["a", "b"], "b", k=5) == 1
    assert recall_at_k(["x"], "b", k=5) == 0


def test_reciprocal_rank() -> None:
    """Test reciprocal rank metric values."""
    assert reciprocal_rank(["a", "b", "c"], "a") == 1.0
    assert reciprocal_rank(["a", "b", "c"], "b") == 0.5
    assert reciprocal_rank(["a", "b", "c"], "z") == 0.0