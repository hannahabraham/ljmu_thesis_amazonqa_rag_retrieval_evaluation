"""Additional coverage for deterministic evaluation and utility helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from src.evaluation.faithfulness import (
    aggregate_groundedness,
    aggregate_hallucination_rate,
    groundedness,
    hallucination_rate_row,
)
from src.evaluation.latency import avg_latency_per_question_ms, latency_detail
from src.evaluation.robustness import (
    long_context_metrics,
    noise_robustness_metrics,
)
from src.llm_clients.error_terms import (
    is_daily_quota_error,
    should_rotate_key,
    should_try_next_key,
)
from src.utils import caching
from src.utils.io import load_per_question, parse_list_field, read_jsonl, write_jsonl


def test_lexical_faithfulness_metrics_handle_supported_and_empty_rows() -> None:
    """Test groundedness and hallucination helpers on supported and skipped rows."""
    assert groundedness("Battery lasts eight hours", ["The battery lasts hours"]) == 0.75
    assert hallucination_rate_row("Battery lasts eight hours", ["battery lasts"]) == 0.5
    assert math.isnan(groundedness("", ["anything"]))

    answers = ["Battery lasts", "I do not know", ""]
    contexts = [["battery lasts"], ["battery"], ["empty"]]
    refused = [False, True, False]

    assert aggregate_groundedness(answers, contexts, refused) == 1.0
    assert aggregate_hallucination_rate(answers, contexts, refused) == 0.0


def test_lexical_aggregates_reject_misaligned_inputs() -> None:
    """Test aggregate helpers validate input lengths."""
    try:
        aggregate_groundedness(["a"], [["a"]], [False, True])
    except ValueError as error:
        assert "align" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("expected aggregate_groundedness to reject lengths")

    try:
        aggregate_hallucination_rate(["a"], [["a"]], [False, True])
    except ValueError as error:
        assert "align" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("expected aggregate_hallucination_rate to reject lengths")


def test_latency_metrics_cover_total_fallback_and_empty_frames() -> None:
    """Test latency averages and percentiles for total and component columns."""
    frame = pd.DataFrame(
        {
            "retrieval_ms": [10, 20, 30],
            "generation_ms": [90, 80, 70],
        }
    )

    assert avg_latency_per_question_ms(frame) == 100.0

    detail = latency_detail(frame)
    assert detail["retrieval_p50_ms"] == 20.0
    assert detail["generation_p50_ms"] == 80.0
    assert detail["total_p50_ms"] == 100.0

    assert math.isnan(avg_latency_per_question_ms(pd.DataFrame()))
    assert math.isnan(latency_detail(pd.DataFrame())["total_p95_ms"])


def _robustness_frame() -> pd.DataFrame:
    """Return a small per-question frame with long questions and noisy retrieval."""
    return pd.DataFrame(
        [
            {
                "question": " ".join(["word"] * 14),
                "q_bucket": "long",
                "gold_answer": "alpha beta",
                "generated_answer": "alpha beta",
                "is_answerable": 1,
                "refused": False,
                "evidence_doc_id": "d1",
                "retrieved_doc_ids": ["d1", "d2", "d3"],
            },
            {
                "question": " ".join(["word"] * 14),
                "q_bucket": "long",
                "gold_answer": "[UNANSWERABLE]",
                "generated_answer": "",
                "is_answerable": 0,
                "refused": True,
                "evidence_doc_id": "d2",
                "retrieved_doc_ids": ["d9", "d2", "d3"],
            },
            {
                "question": "short question",
                "q_bucket": "short",
                "gold_answer": "gamma",
                "generated_answer": "wrong",
                "is_answerable": 1,
                "refused": False,
                "evidence_doc_id": "d3",
                "retrieved_doc_ids": ["d9", "d8", "d3"],
            },
        ]
    )


def test_robustness_metrics_cover_long_and_noisy_slices() -> None:
    """Test long-context and noise robustness calculations."""
    frame = _robustness_frame()

    long_metrics = long_context_metrics(frame)
    assert long_metrics["long_context_n"] == 2
    assert long_metrics["long_context_f1"] == 1.0
    assert long_metrics["long_context_answerability"] == 1.0

    noise_metrics = noise_robustness_metrics(frame, k=3)
    assert noise_metrics["noise_n"] >= 1
    assert "noise_robust_f1" in noise_metrics


def test_io_list_jsonl_and_per_question_loading(tmp_path: Path) -> None:
    """Test list parsing, JSONL round-trip, and filename-based filtering."""
    assert parse_list_field(None) == []
    assert parse_list_field("[1, 2]") == [1, 2]
    assert parse_list_field("array([1, 2], dtype=int64)") == [1, 2]
    assert parse_list_field(("a", "b")) == ["a", "b"]
    assert parse_list_field({"x": 1}) == [{"x": 1}]

    jsonl_path = tmp_path / "per_question" / "bm25_k5_seed42.jsonl"
    rows = [{"pipeline": "bm25", "k": 5, "value": 1}]
    assert write_jsonl(rows, jsonl_path) == 1
    assert read_jsonl(jsonl_path) == rows

    loaded = load_per_question(
        tmp_path / "per_question",
        pipelines=["bm25"],
        ks=[5],
        seed=42,
    )
    assert loaded.to_dict("records") == rows
    assert load_per_question(tmp_path / "missing").empty


def test_disk_cache_round_trip_and_bad_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Test cache get/set behaviour and corrupt cache recovery."""
    monkeypatch.setattr(caching, "LLM_CACHE_DIR", tmp_path)

    assert caching.get_cached("unit", "missing") is None
    caching.set_cached("unit", {"score": 1}, "question", "answer")
    assert caching.get_cached("unit", "question", "answer") == {"score": 1}

    bad_path = caching.cache_path("unit", "bad")
    bad_path.write_text("{not-json", encoding="utf-8")
    assert caching.get_cached("unit", "bad") is None


def test_error_term_classification() -> None:
    """Test daily quota, retry, and rotation error classification."""
    daily = RuntimeError("429 quota exceeded: requests per day for project")
    transient = RuntimeError("503 overloaded")
    auth = RuntimeError("403 permission denied")
    ordinary = RuntimeError("plain validation error")

    assert is_daily_quota_error(daily) is True
    assert should_try_next_key(daily) is False
    assert should_rotate_key(daily) is False
    assert should_try_next_key(transient) is True
    assert should_rotate_key(transient) is False
    assert should_rotate_key(auth) is True
    assert should_try_next_key(ordinary) is False
