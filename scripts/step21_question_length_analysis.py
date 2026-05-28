"""Compute question-length bucket F1 and answerability metrics for all k values."""

from __future__ import annotations

import logging

import pandas as pd

from config.settings import (
    K_VALUES,
    OUTPUT_DIR,
    PER_QUESTION_DIR,
    PIPELINE_KEYS,
    RANDOM_SEED,
)
from src.evaluation.answerability import compute_answerability_table
from src.evaluation.generation_metrics import token_f1
from src.evaluation.statistics import bootstrap_ci, is_indicative
from src.sampling import assign_q_bucket
from src.utils.io import load_per_question
from src.utils.logging_config import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)


def _load_answers(pipeline: str) -> pd.DataFrame:
    """Load per-question rows for one pipeline across all k values."""
    dataframe = load_per_question(
        PER_QUESTION_DIR,
        pipelines=[pipeline],
        ks=list(K_VALUES),
        seed=RANDOM_SEED,
    )

    if dataframe.empty:
        LOGGER.warning("Missing per-question rows for %s", pipeline)
    return dataframe.copy()


def _ensure_question_bucket(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Ensure q_bucket exists, deriving it from question text if needed."""
    if "q_bucket" not in dataframe.columns or dataframe["q_bucket"].isna().all():
        dataframe["q_bucket"] = dataframe["question"].apply(assign_q_bucket)

    return dataframe


def _compute_pipeline_rows(
    pipeline: str,
    dataframe: pd.DataFrame,
) -> list[dict]:
    """Compute bucket-level F1 and answerability rows for one pipeline."""
    rows: list[dict] = []

    answerable = dataframe[
        dataframe["gold_answer"].astype(str).str.upper() != "[UNANSWERABLE]"
    ].copy()
    answerable["f1"] = [
        token_f1(row["generated_answer"], row["gold_answer"])
        for _, row in answerable.iterrows()
    ]

    for (k_value, bucket), group in dataframe.groupby(["k", "q_bucket"]):
        answerable_group = answerable[
            (answerable["k"] == k_value)
            & (answerable["q_bucket"] == bucket)
        ]
        f1_mean, lower_bound, upper_bound = bootstrap_ci(
            answerable_group["f1"].tolist()
        )

        answerability_table = compute_answerability_table(
            group.assign(is_answerable=group["is_answerable"].astype(int))
        )

        rows.append(
            {
                "pipeline": pipeline,
                "k": int(k_value),
                "q_bucket": bucket,
                "n": len(group),
                "f1": f1_mean,
                "f1_lo": lower_bound,
                "f1_hi": upper_bound,
                "answerability_acc": float(
                    answerability_table["answerability_acc"].iloc[0]
                ),
                "indicative": is_indicative(len(group)),
            }
        )

    return rows


def main() -> None:
    """Build and save question-bucket metrics for all pipelines."""
    rows: list[dict] = []

    for pipeline in PIPELINE_KEYS:
        dataframe = _load_answers(pipeline)

        if dataframe.empty:
            continue

        dataframe = _ensure_question_bucket(dataframe)

        rows.extend(
            _compute_pipeline_rows(
                pipeline,
                dataframe,
            )
        )

    output_path = OUTPUT_DIR / "qbucket_metrics.csv"

    pd.DataFrame(rows).to_csv(output_path, index=False)

    LOGGER.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
