"""Tests for EDA and pipeline runner logic without external services."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import eda
from src.pipelines import runner
from src.utils.io import read_jsonl


def _answers_df() -> pd.DataFrame:
    """Return a tiny generated-answer frame for runner metric tests."""
    return pd.DataFrame(
        [
            {
                "golden_id": "g1",
                "record_id": "r1",
                "asin": "a1",
                "category": "Electronics",
                "q_bucket": "long",
                "question_type": "yesno",
                "pipeline": "bm25",
                "k": 2,
                "question": " ".join(["word"] * 14),
                "gold_answer": "yes waterproof",
                "evidence_doc_id": "d1",
                "is_answerable": 1,
                "retrieved_doc_ids": json.dumps(["d1", "d2"]),
                "retrieved_context": json.dumps(["yes waterproof context", "noise"]),
                "retrieval_ms": 10.0,
                "generated_answer": "yes waterproof",
                "refused": False,
                "generation_ms": 90.0,
            },
            {
                "golden_id": "g2",
                "record_id": "r2",
                "asin": "a2",
                "category": "Toys_and_Games",
                "q_bucket": "short",
                "question_type": "open",
                "pipeline": "bm25",
                "k": 2,
                "question": "short question",
                "gold_answer": "[UNANSWERABLE]",
                "evidence_doc_id": None,
                "is_answerable": 0,
                "retrieved_doc_ids": json.dumps(["d9"]),
                "retrieved_context": json.dumps(["noise"]),
                "retrieval_ms": 20.0,
                "generated_answer": "",
                "refused": True,
                "generation_ms": 80.0,
            },
        ]
    )


def test_eda_summary_and_plots(tmp_path: Path) -> None:
    """Test EDA summary and plot generation on local synthetic data."""
    frame = pd.DataFrame(
        [
            {
                "qid": "q1",
                "is_answerable": 1,
                "questionType": "yesno",
                "answers": [
                    {"helpful": [2, 3]},
                    {"helpful": 1, "unhelpful": 1},
                ],
                "category": "Electronics",
                "n_snippets": 3,
                "n_answers": 2,
            },
            {
                "qid": "q1",
                "is_answerable": 0,
                "questionType": "open",
                "answers": [],
                "category": "Toys_and_Games",
                "n_snippets": 1,
                "n_answers": 0,
            },
        ]
    )

    summary = eda.summarise_split(frame, "unit")
    eda.make_plots(frame, "unit", out_dir=tmp_path)
    eda.make_plots(pd.DataFrame(), "empty", out_dir=tmp_path)

    assert summary["n_rows"] == 2
    assert summary["n_duplicates_qid"] == 1
    assert summary["votes_zero"] == 1
    assert (tmp_path / "unit_vote_distribution.png").exists()
    assert (tmp_path / "unit_dataset_profile.png").exists()


def test_runner_metrics_jsonl_and_upserts(tmp_path: Path, monkeypatch) -> None:
    """Test runner metric computation, JSONL writing, and CSV upserts."""
    monkeypatch.setattr(runner, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        runner,
        "pipeline_output_dir",
        lambda pipeline: tmp_path / pipeline,
    )
    (tmp_path / "bm25").mkdir()

    answers = _answers_df()
    metrics = runner.compute_per_cell_metrics(answers, k=2)

    assert metrics["Total Questions"] == 2
    assert metrics["Recall@K"] == 1.0
    assert metrics["Avg Latency / Question (s)"] == 0.1

    jsonl_path = runner._write_per_question_jsonl(
        answers,
        "bm25",
        2,
        seed=42,
        output_dir=tmp_path / "per_question",
    )
    rows = read_jsonl(jsonl_path)
    assert rows[0]["run_id"] == "bm25_k2_seed42"
    assert rows[0]["total_ms"] == 100.0

    runner._upsert_pipeline_summary("bm25", 2, metrics)
    runner.upsert_results_row("bm25", 2, metrics)

    assert (tmp_path / "bm25" / "summary.csv").exists()
    assert (tmp_path / "results.csv").exists()


def test_runner_generation_uses_cache_and_live_client(tmp_path: Path, monkeypatch) -> None:
    """Test generation path with one cached answer and one fake live answer."""
    monkeypatch.setattr(
        runner,
        "pipeline_output_dir",
        lambda pipeline: tmp_path / pipeline,
    )
    (tmp_path / "bm25").mkdir()

    retrieval = _answers_df().drop(columns=["generated_answer", "refused", "generation_ms"])
    retrieval.loc[0, "retrieved_doc_ids"] = json.dumps(["d1"])
    retrieval.loc[0, "retrieved_context"] = json.dumps(["context one"])
    retrieval.loc[1, "retrieved_doc_ids"] = json.dumps(["d2"])
    retrieval.loc[1, "retrieved_context"] = json.dumps(["context two"])

    cache_hits = iter(
        [
            {"generated_answer": "cached answer", "generation_ms": 5.0},
            None,
        ]
    )
    monkeypatch.setattr(runner, "get_cached", lambda *args: next(cache_hits))
    monkeypatch.setattr(runner, "set_cached", lambda *args: None)
    monkeypatch.setattr(runner, "load_groq_keys", lambda: ["key"])

    class _FakeGroq:
        def __init__(self, **kwargs):
            pass

        def batch_invoke(self, prompts):
            return ["live answer"], [7.0]

    monkeypatch.setattr(runner, "GroqClient", _FakeGroq)

    generated = runner._run_generation(retrieval, "bm25", 2)

    assert generated["generated_answer"].tolist() == ["cached answer", "live answer"]
    assert generated["generation_ms"].tolist() == [5.0, 7.0]


def test_run_pipeline_cell_with_fake_steps(tmp_path: Path, monkeypatch) -> None:
    """Test top-level cell runner using fake retrieval/generation steps."""
    answers = _answers_df()

    monkeypatch.setattr(runner, "_run_retrieval", lambda pipeline, k, sample: answers)
    monkeypatch.setattr(runner, "_run_generation", lambda retrieval_df, pipeline, k: answers)
    monkeypatch.setattr(runner, "_upsert_pipeline_summary", lambda *args: None)
    monkeypatch.setattr(runner, "upsert_results_row", lambda *args: None)

    metrics = runner.run_pipeline_cell(
        "bm25",
        2,
        sample=1,
        seed=42,
        output_dir=tmp_path,
    )

    assert metrics["Total Questions"] == 2
    assert (tmp_path / "bm25_k2_seed42.jsonl").exists()

    try:
        runner.run_pipeline_cell("unknown", 2, output_dir=tmp_path)
    except ValueError as error:
        assert "unknown pipeline" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("unknown pipeline should raise")
