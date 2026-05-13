"""Composite ranking (EXP_20) + paired Wilcoxon block + sensitivity sweeps.

Sources of truth: per-question JSONL in outputs/per_question/ (k=5) + the
RAGAS aggregate in outputs/ragas_metrics.csv. The Table 7 builder handles the
composite formula and best-k argmax; this script also writes:

  outputs/tables/pairwise_wilcoxon.csv   — paired Wilcoxon on token_f1 + faithfulness
  outputs/correct_threshold_sensitivity.csv  — Correct Answers at thresholds 0.3/0.5/0.7
  outputs/composite_weight_sensitivity.csv  — ranking under alternate weight vectors
"""
from __future__ import annotations

import logging

import pandas as pd

from config.settings import (
    CORRECT_F1_SENSITIVITY,
    OUTPUT_DIR,
    PER_QUESTION_DIR,
    PIPELINE_KEYS,
    RANDOM_SEED,
    TABLES_DIR,
)
from src.evaluation.generation_metrics import correct_answers_count
from src.evaluation.table_builders import (
    COMPOSITE_WEIGHTS,
    PIPELINE_LABEL,
    build_pairwise_wilcoxon,
    build_table7_final_ranking,
)
from src.utils.io import load_per_question
from src.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


ALTERNATE_WEIGHT_VECTORS: dict[str, dict[str, float]] = {
    "primary": COMPOSITE_WEIGHTS,
    "f1_heavy": {
        "f1": 0.40, "faithfulness": 0.20, "context_precision": 0.10,
        "context_recall": 0.05, "answerability_acc": 0.15,
        "category_consistency": 0.05, "latency": 0.05,
    },
    "faithfulness_heavy": {
        "f1": 0.15, "faithfulness": 0.35, "context_precision": 0.15,
        "context_recall": 0.10, "answerability_acc": 0.15,
        "category_consistency": 0.05, "latency": 0.05,
    },
    "balanced_quality": {
        "f1": 0.25, "faithfulness": 0.25, "context_precision": 0.15,
        "context_recall": 0.15, "answerability_acc": 0.15,
        "category_consistency": 0.00, "latency": 0.05,
    },
}


def _load_full_per_question() -> pd.DataFrame:
    df = load_per_question(
        PER_QUESTION_DIR, pipelines=list(PIPELINE_KEYS), seed=RANDOM_SEED,
    )
    if df.empty:
        raise SystemExit(f"No per-question JSONL found in {PER_QUESTION_DIR}")
    return df


def _attach_ragas(per_q: pd.DataFrame) -> pd.DataFrame:
    path = OUTPUT_DIR / "ragas_metrics.csv"
    if not path.exists():
        return per_q
    ragas = pd.read_csv(path)
    return per_q.merge(
        ragas.rename(columns={
            "faithfulness": "_agg_faithfulness",
            "context_precision": "_agg_context_precision",
            "context_recall": "_agg_context_recall",
        })[["pipeline", "k", "_agg_faithfulness", "_agg_context_precision", "_agg_context_recall"]],
        on=["pipeline", "k"], how="left",
    )


def _write_correct_threshold_sensitivity(per_q: pd.DataFrame) -> None:
    """Recompute Correct Answers at thresholds 0.3 / 0.5 / 0.7 (k=5)."""
    k5 = per_q[per_q["k"] == 5]
    rows: list[dict] = []
    for pipeline in PIPELINE_KEYS:
        sub = k5[k5["pipeline"] == pipeline]
        if sub.empty:
            continue
        for threshold in CORRECT_F1_SENSITIVITY:
            rows.append({
                "pipeline": pipeline,
                "pipeline_label": PIPELINE_LABEL.get(pipeline, pipeline),
                "f1_threshold": threshold,
                "correct_answers": correct_answers_count(sub, threshold),
                "total_questions": int(len(sub)),
            })
    df = pd.DataFrame(rows)
    out = OUTPUT_DIR / "correct_threshold_sensitivity.csv"
    df.to_csv(out, index=False)
    logger.info("Wrote %s", out)


def _write_weight_sensitivity(per_q: pd.DataFrame) -> None:
    """Recompute Table 7 composite under several weight vectors."""
    rows: list[dict] = []
    for name, weights in ALTERNATE_WEIGHT_VECTORS.items():
        table = build_table7_final_ranking(per_q, weights=weights)
        for _, r in table.iterrows():
            rows.append({
                "weight_vector": name,
                "pipeline": r["Pipeline"],
                "best_k": r["Best K"],
                "composite_score": r["Composite Score"],
                "rank": r["Rank"],
            })
    df = pd.DataFrame(rows)
    out = OUTPUT_DIR / "composite_weight_sensitivity.csv"
    df.to_csv(out, index=False)
    logger.info("Wrote %s", out)

    # Surface whether the top-ranked pipeline changes
    primary_top = df[df["weight_vector"] == "primary"].sort_values("rank")["pipeline"].iloc[0]
    for name in ALTERNATE_WEIGHT_VECTORS:
        if name == "primary":
            continue
        alt_top = df[df["weight_vector"] == name].sort_values("rank")["pipeline"].iloc[0]
        if alt_top != primary_top:
            logger.warning(
                "Sensitivity: ranking changes under %s (top = %s, primary top = %s)",
                name, alt_top, primary_top,
            )


def main() -> None:
    per_q = _load_full_per_question()
    per_q = _attach_ragas(per_q)

    # Use per-row RAGAS when available; otherwise broadcast the aggregate.
    for col in ("faithfulness", "context_precision", "context_recall"):
        if col not in per_q.columns:
            per_q[col] = pd.NA
        agg_col = f"_agg_{col}"
        if agg_col in per_q.columns:
            per_q[col] = pd.to_numeric(per_q[col], errors="coerce").fillna(per_q[agg_col])

    table7 = build_table7_final_ranking(per_q)
    table7_path = TABLES_DIR / "table7_final_ranking.csv"
    table7.to_csv(table7_path, index=False)
    logger.info("Wrote %s (%d rows)", table7_path, len(table7))

    pairwise = build_pairwise_wilcoxon(per_q, metrics=("token_f1", "faithfulness"))
    pairwise_path = TABLES_DIR / "pairwise_wilcoxon.csv"
    pairwise.to_csv(pairwise_path, index=False)
    logger.info("Wrote %s (%d rows)", pairwise_path, len(pairwise))

    _write_correct_threshold_sensitivity(per_q)
    _write_weight_sensitivity(per_q)


if __name__ == "__main__":
    main()
