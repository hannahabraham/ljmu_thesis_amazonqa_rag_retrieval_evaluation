"""Tests for question-length bucket metric aggregation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts import step21_question_length_analysis as qlength
from src.utils.io import write_jsonl


def test_qbucket_metrics_include_answerability_accuracy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Question buckets should report answerability over all rows, not only F1."""
    per_question_dir = tmp_path / "per_question"
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    write_jsonl(
        [
            {
                "pipeline": "bm25",
                "k": 1,
                "seed": 42,
                "q_bucket": "short",
                "question": "Is it red?",
                "gold_answer": "yes",
                "generated_answer": "yes",
                "is_answerable": True,
                "refused": False,
            },
            {
                "pipeline": "bm25",
                "k": 1,
                "seed": 42,
                "q_bucket": "short",
                "question": "Is warranty included?",
                "gold_answer": "[UNANSWERABLE]",
                "generated_answer": "The available reviews do not say.",
                "is_answerable": False,
                "refused": True,
            },
            {
                "pipeline": "bm25",
                "k": 1,
                "seed": 42,
                "q_bucket": "short",
                "question": "What color is it?",
                "gold_answer": "red",
                "generated_answer": "The available reviews do not say.",
                "is_answerable": True,
                "refused": True,
            },
        ],
        per_question_dir / "bm25_k1_seed42.jsonl",
    )

    monkeypatch.setattr(qlength, "PER_QUESTION_DIR", per_question_dir)
    monkeypatch.setattr(qlength, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(qlength, "PIPELINE_KEYS", ("bm25",))
    monkeypatch.setattr(qlength, "K_VALUES", (1,))
    monkeypatch.setattr(qlength, "RANDOM_SEED", 42)

    qlength.main()

    metrics = pd.read_csv(output_dir / "qbucket_metrics.csv")

    assert list(metrics["q_bucket"]) == ["short"]
    assert metrics["n"].iloc[0] == 3
    assert metrics["f1"].iloc[0] == 0.5
    assert metrics["answerability_acc"].iloc[0] == 2 / 3
