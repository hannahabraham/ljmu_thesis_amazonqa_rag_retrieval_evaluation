"""Category-level F1 + answerability at k=5 (Sheet Table 3).

v5: restricted to the four NAMED_CATEGORIES — records outside that set are
excluded from this analysis (still contribute to Tables 1, 2, 4, 6, 7).
"""
from __future__ import annotations

import logging

import pandas as pd

from config.settings import (
    NAMED_CATEGORIES,
    OUTPUT_DIR,
    PER_QUESTION_DIR,
    PIPELINE_KEYS,
    RANDOM_SEED,
)
from src.evaluation.answerability import compute_answerability_table
from src.evaluation.statistics import bootstrap_ci, is_indicative
from src.utils.io import load_per_question
from src.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    rows: list[dict] = []
    per_q = load_per_question(
        PER_QUESTION_DIR, pipelines=list(PIPELINE_KEYS), ks=[5], seed=RANDOM_SEED,
    )
    if per_q.empty:
        logger.error("No per-question JSONL found at %s for k=5", PER_QUESTION_DIR)
        return

    per_q = per_q[per_q["category"].isin(NAMED_CATEGORIES)].copy()
    per_q["is_answerable"] = per_q["is_answerable"].astype(bool)

    for pipeline in PIPELINE_KEYS:
        sub = per_q[per_q["pipeline"] == pipeline]
        if sub.empty:
            continue
        for category, group in sub.groupby("category"):
            answerable = group[group["is_answerable"]]
            f1_vals = pd.to_numeric(answerable.get("token_f1", pd.Series(dtype=float)),
                                    errors="coerce").dropna().tolist()
            f1_mean, lo, hi = bootstrap_ci(f1_vals) if f1_vals else (float("nan"),) * 3
            ans_table = compute_answerability_table(
                group.assign(is_answerable=group["is_answerable"].astype(int))
            )
            rows.append({
                "pipeline": pipeline,
                "category": category,
                "n": len(group),
                "f1": f1_mean, "f1_lo": lo, "f1_hi": hi,
                "answerability_acc": float(ans_table["answerability_acc"].iloc[0]),
                "indicative": is_indicative(len(group)),
            })

    out = OUTPUT_DIR / "category_metrics.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info("Wrote %s (%d rows)", out, len(rows))


if __name__ == "__main__":
    main()
