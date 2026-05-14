"""Compute category-level F1 and answerability metrics at k=5.

Only categories in ``NAMED_CATEGORIES`` are included in this analysis.
Rows outside that set still contribute to the other result tables.
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
LOGGER = logging.getLogger(__name__)


def _load_named_category_rows() -> pd.DataFrame:
    """Load k=5 per-question rows restricted to named categories."""
    per_question = load_per_question(
        PER_QUESTION_DIR,
        pipelines=list(PIPELINE_KEYS),
        ks=[5],
        seed=RANDOM_SEED,
    )

    if per_question.empty:
        LOGGER.error(
            "No per-question JSONL found at %s for k=5",
            PER_QUESTION_DIR,
        )
        return pd.DataFrame()

    per_question = per_question[
        per_question["category"].isin(NAMED_CATEGORIES)
    ].copy()

    per_question["is_answerable"] = (
        per_question["is_answerable"].astype(bool)
    )

    return per_question


def _answerable_f1_ci(group: pd.DataFrame) -> tuple[float, float, float]:
    """Compute bootstrap CI for answerable-row token F1."""
    answerable = group[group["is_answerable"]]

    f1_values = (
        pd.to_numeric(
            answerable.get("token_f1", pd.Series(dtype=float)),
            errors="coerce",
        )
        .dropna()
        .tolist()
    )

    if not f1_values:
        return (float("nan"), float("nan"), float("nan"))

    return bootstrap_ci(f1_values)


def _answerability_accuracy(group: pd.DataFrame) -> float:
    """Compute answerability accuracy for a grouped subset."""
    answerability_table = compute_answerability_table(
        group.assign(
            is_answerable=group["is_answerable"].astype(int)
        )
    )

    return float(answerability_table["answerability_acc"].iloc[0])


def main() -> None:
    """Build and save category-level metrics for all pipelines."""
    rows: list[dict] = []

    per_question = _load_named_category_rows()

    if per_question.empty:
        return

    for pipeline in PIPELINE_KEYS:
        pipeline_rows = per_question[
            per_question["pipeline"] == pipeline
        ]

        if pipeline_rows.empty:
            continue

        for category, group in pipeline_rows.groupby("category"):
            f1_mean, lower_bound, upper_bound = _answerable_f1_ci(group)

            rows.append(
                {
                    "pipeline": pipeline,
                    "category": category,
                    "n": len(group),
                    "f1": f1_mean,
                    "f1_lo": lower_bound,
                    "f1_hi": upper_bound,
                    "answerability_acc": _answerability_accuracy(group),
                    "indicative": is_indicative(len(group)),
                }
            )

    output_path = OUTPUT_DIR / "category_metrics.csv"

    pd.DataFrame(rows).to_csv(output_path, index=False)

    LOGGER.info(
        "Wrote %s (%d rows)",
        output_path,
        len(rows),
    )


if __name__ == "__main__":
    main()