"""Answerability accuracy + long-context + noise-robustness per (pipeline, k).

Reads per-pipeline answers CSVs and writes:
  - outputs/answerability_metrics.csv          (cross-pipeline)
  - outputs/<pipeline>/answerability_metrics.csv (per-pipeline mirror)
"""
from __future__ import annotations

import logging
from itertools import product

import pandas as pd

from config.settings import K_VALUES, OUTPUT_DIR, PIPELINE_KEYS, pipeline_output_dir
from src.evaluation.answerability import compute_answerability_table
from src.evaluation.robustness import long_context_metrics, noise_robustness_metrics
from src.evaluation.statistics import wilson_ci
from src.utils.io import parse_list_field
from src.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    rows: list[dict] = []
    for pipeline, k in product(PIPELINE_KEYS, K_VALUES):
        path = pipeline_output_dir(pipeline) / f"answers_k{k}.csv"
        if not path.exists():
            continue
        full = pd.read_csv(path)
        full["retrieved_doc_ids"] = full["retrieved_doc_ids"].apply(parse_list_field)
        full["retrieved_context"] = full["retrieved_context"].apply(parse_list_field)

        df = full[full["is_answerable"].notna()].copy()
        df["is_answerable"] = df["is_answerable"].astype(int)
        if df.empty:
            continue

        table = compute_answerability_table(df)
        n = int(table["n"].iloc[0])
        successes = int(table["correctly_answered"].iloc[0]) + int(table["correctly_refused"].iloc[0])
        acc, lo, hi = wilson_ci(successes, n)

        long_ctx = long_context_metrics(full)
        noise = noise_robustness_metrics(full, k)

        row = {
            "pipeline": pipeline, "k": k, "n": n,
            "correctly_answered": int(table["correctly_answered"].iloc[0]),
            "wrongly_refused": int(table["wrongly_refused"].iloc[0]),
            "correctly_refused": int(table["correctly_refused"].iloc[0]),
            "wrongly_answered": int(table["wrongly_answered"].iloc[0]),
            "answerability_acc": acc,
            "answerability_acc_lo": lo, "answerability_acc_hi": hi,
            **long_ctx,
            **noise,
        }
        rows.append(row)

        # Mirror per-pipeline
        per_pipeline_path = pipeline_output_dir(pipeline) / "answerability_metrics.csv"
        existing = (
            pd.read_csv(per_pipeline_path) if per_pipeline_path.exists() else pd.DataFrame()
        )
        existing = existing[~((existing.get("pipeline") == pipeline) & (existing.get("k") == k))] \
            if not existing.empty else existing
        out_df = pd.concat([existing, pd.DataFrame([row])], ignore_index=True) \
            .sort_values("k").reset_index(drop=True)
        out_df.to_csv(per_pipeline_path, index=False)

    out = OUTPUT_DIR / "answerability_metrics.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
