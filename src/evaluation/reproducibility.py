"""Reproducibility drift across two seeded runs (v5).

The thesis runs all 5 pipelines at k=5 a second time with REPRO_SEED_2 and
reports |Δ| on F1, faithfulness, and answerability_accuracy. Drift > 2%
absolute on F1 is flagged in the thesis limitations chapter — `temperature=0`
makes the generator deterministic in principle but Groq's serving stack can
still vary.
"""
from __future__ import annotations

import pandas as pd

from src.evaluation.answerability import compute_answerability_table

METRICS_FOR_DRIFT = ("token_f1", "faithfulness", "answerability_accuracy")
F1_DRIFT_FLAG_THRESHOLD = 0.02


def _per_pipeline_aggregate(per_q: pd.DataFrame) -> dict[str, float]:
    """Compute the three drift metrics from a per-question DataFrame."""
    token_f1 = pd.to_numeric(per_q.get("token_f1"), errors="coerce").dropna()
    faithfulness = pd.to_numeric(per_q.get("faithfulness"), errors="coerce").dropna()

    df = per_q.copy()
    if "is_answerable" in df.columns:
        df["is_answerable"] = df["is_answerable"].apply(
            lambda v: int(bool(v)) if pd.notna(v) else None
        )
        df = df[df["is_answerable"].notna()]

    answerability = (
        float(compute_answerability_table(df)["answerability_acc"].iloc[0])
        if len(df) else float("nan")
    )
    return {
        "token_f1": float(token_f1.mean()) if not token_f1.empty else float("nan"),
        "faithfulness": float(faithfulness.mean()) if not faithfulness.empty else float("nan"),
        "answerability_accuracy": answerability,
    }


def reproducibility_drift(
    seed1_df: pd.DataFrame, seed2_df: pd.DataFrame,
) -> pd.DataFrame:
    """For each pipeline, report mean values under both seeds and absolute drift.

    Both DataFrames must contain a ``pipeline`` column and per-question metric
    columns. The ``flagged`` boolean marks rows where the F1 drift exceeds
    ``F1_DRIFT_FLAG_THRESHOLD``.
    """
    rows: list[dict] = []
    pipelines = sorted(
        set(seed1_df["pipeline"].unique()) | set(seed2_df["pipeline"].unique())
    )
    for pipeline in pipelines:
        s1 = seed1_df[seed1_df["pipeline"] == pipeline]
        s2 = seed2_df[seed2_df["pipeline"] == pipeline]
        agg1 = _per_pipeline_aggregate(s1)
        agg2 = _per_pipeline_aggregate(s2)
        for metric in METRICS_FOR_DRIFT:
            m1, m2 = agg1[metric], agg2[metric]
            drift = abs(m1 - m2) if pd.notna(m1) and pd.notna(m2) else float("nan")
            denom = max(abs(m1), 1e-9)
            drift_pct = drift / denom if pd.notna(drift) else float("nan")
            flagged = bool(
                metric == "token_f1"
                and pd.notna(drift)
                and drift > F1_DRIFT_FLAG_THRESHOLD
            )
            rows.append({
                "pipeline": pipeline,
                "metric": metric,
                "seed1_mean": m1,
                "seed2_mean": m2,
                "abs_drift": drift,
                "drift_pct": drift_pct,
                "flagged": flagged,
            })
    return pd.DataFrame(rows)
