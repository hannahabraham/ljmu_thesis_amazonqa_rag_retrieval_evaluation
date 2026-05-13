"""Full re-aggregation of outputs/results.csv from per-cell artefacts.

Reads every answers_<pipeline>_k<k>.csv that exists, recomputes the per-cell
metrics, merges in RAGAS metrics from outputs/ragas_metrics.csv (if present),
computes a composite Rank across rows, and writes outputs/results.csv.

Run this:
  - after scripts/18_eval_ragas.py to merge RAGAS into the live table
  - any time you want a freshly-ranked snapshot

Composite ranking:
  0.30*F1 + 0.20*Faithfulness + 0.15*ContextPrecision + 0.10*ContextRecall
  + 0.20*AnswerabilityAcc + 0.05*(1 - normalised_latency)

Missing RAGAS values count as 0 in the composite; the underlying cells stay
blank. Cells with n<INDICATIVE_THRESHOLD are not flagged here -- see
scripts/22_final_ranking.py for the indicative-aware composite.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config.settings import K_VALUES, OUTPUT_DIR, PIPELINE_KEYS, pipeline_output_dir
from src.pipelines.runner import (
    PIPELINE_LABEL,
    RESULTS_COLUMNS,
    compute_per_cell_metrics,
)
from src.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

PIPELINES = PIPELINE_KEYS

WEIGHTS = {
    "f1": 0.30,
    "faithfulness": 0.20,
    "context_precision": 0.15,
    "context_recall": 0.10,
    "answerability_acc": 0.20,
    "latency": 0.05,
}


def _load_ragas() -> pd.DataFrame:
    path = OUTPUT_DIR / "ragas_metrics.csv"
    if not path.exists():
        logger.warning("RAGAS metrics CSV not found at %s; faithfulness/context-* will be blank", path)
        return pd.DataFrame(columns=["pipeline", "k", "faithfulness", "context_precision", "context_recall"])
    return pd.read_csv(path)


def _to_float(value: object) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _composite(row: dict, lat_norm: float) -> float:
    def _coerce(value: float) -> float:
        return 0.0 if np.isnan(value) else float(value)

    return (
        WEIGHTS["f1"] * _coerce(row["f1"])
        + WEIGHTS["faithfulness"] * _coerce(row["faithfulness"])
        + WEIGHTS["context_precision"] * _coerce(row["context_precision"])
        + WEIGHTS["context_recall"] * _coerce(row["context_recall"])
        + WEIGHTS["answerability_acc"] * _coerce(row["answerability_acc"])
        + WEIGHTS["latency"] * (1.0 - lat_norm if not np.isnan(lat_norm) else 0.0)
    )


def main() -> None:
    ragas_df = _load_ragas()

    rows: list[dict] = []
    for pipeline in PIPELINES:
        for k in K_VALUES:
            answers_path = pipeline_output_dir(pipeline) / f"answers_k{k}.csv"
            if not answers_path.exists():
                logger.warning("Missing %s, skipping cell", answers_path)
                continue
            answers = pd.read_csv(answers_path)
            metrics = compute_per_cell_metrics(answers, k)

            ragas_row = ragas_df[(ragas_df["pipeline"] == pipeline) & (ragas_df["k"] == k)]
            faith = _to_float(ragas_row["faithfulness"].iloc[0]) if not ragas_row.empty else float("nan")
            cprec = _to_float(ragas_row["context_precision"].iloc[0]) if not ragas_row.empty else float("nan")
            crec = _to_float(ragas_row["context_recall"].iloc[0]) if not ragas_row.empty else float("nan")

            rows.append({
                "Pipeline": PIPELINE_LABEL[pipeline],
                "pipeline_key": pipeline,
                "K Value": int(k),
                **metrics,
                "Faithfulness Score": round(faith, 4) if not np.isnan(faith) else "",
                "Context Precision": round(cprec, 4) if not np.isnan(cprec) else "",
                "Context Recall": round(crec, 4) if not np.isnan(crec) else "",
                "_f1": _to_float(metrics["F1 Score"]),
                "_faith": faith,
                "_cprec": cprec,
                "_crec": crec,
                "_answer_acc": _to_float(metrics["Answerability Accuracy"]),
                "_avg_latency_s": _to_float(metrics["Avg Latency / Question (s)"]),
            })

    if not rows:
        logger.error("No answers_*.csv found under %s. Run a pipeline first.", OUTPUT_DIR)
        return

    df = pd.DataFrame(rows)

    lat_min = df["_avg_latency_s"].min()
    lat_max = df["_avg_latency_s"].max()
    if lat_max > lat_min:
        lat_norm = (df["_avg_latency_s"] - lat_min) / (lat_max - lat_min)
    else:
        lat_norm = pd.Series([0.0] * len(df), index=df.index)

    composites: list[float] = []
    for i, r in df.iterrows():
        composites.append(_composite(
            {"f1": r["_f1"], "faithfulness": r["_faith"],
             "context_precision": r["_cprec"], "context_recall": r["_crec"],
             "answerability_acc": r["_answer_acc"]},
            float(lat_norm.iloc[i]),
        ))
    df["_composite"] = composites
    df["Rank"] = df["_composite"].rank(method="min", ascending=False).astype(int)

    out_columns = list(RESULTS_COLUMNS) + ["Rank"]
    out_df = (
        df.sort_values(["Rank", "Pipeline", "K Value"])
        .reindex(columns=out_columns)
        .reset_index(drop=True)
    )

    out_path = OUTPUT_DIR / "results.csv"
    out_df.to_csv(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(out_df))

    # Mirror the cross-pipeline table back into each pipeline's folder so the
    # per-pipeline summary.csv reflects merged RAGAS scores too.
    for pipeline in PIPELINES:
        sub = out_df[out_df["pipeline_key"] == pipeline]
        if sub.empty:
            continue
        sub_path = pipeline_output_dir(pipeline) / "summary.csv"
        sub.reindex(columns=RESULTS_COLUMNS).to_csv(sub_path, index=False)


if __name__ == "__main__":
    main()
