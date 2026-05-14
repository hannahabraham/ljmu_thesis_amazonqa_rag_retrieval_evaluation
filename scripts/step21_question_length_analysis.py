"""Compute question-length bucket F1 metrics at k=5."""

from __future__ import annotations

import logging

import pandas as pd

from config.settings import OUTPUT_DIR, PIPELINE_KEYS, pipeline_output_dir
from src.evaluation.generation_metrics import token_f1
from src.evaluation.statistics import bootstrap_ci, is_indicative
from src.sampling import assign_q_bucket
from src.utils.logging_config import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)


def _load_answerable_answers(pipeline: str) -> pd.DataFrame:
    """Load answerable k=5 answer rows for one pipeline."""
    answers_path = pipeline_output_dir(pipeline) / "answers_k5.csv"

    if not answers_path.exists():
        LOGGER.warning("Missing answers file: %s", answers_path)
        return pd.DataFrame()

    dataframe = pd.read_csv(answers_path)

    return dataframe[
        dataframe["gold_answer"].astype(str).str.upper() != "[UNANSWERABLE]"
    ].copy()


def _ensure_question_bucket(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Ensure q_bucket exists, deriving it from question text if needed."""
    if "q_bucket" not in dataframe.columns or dataframe["q_bucket"].isna().all():
        dataframe["q_bucket"] = dataframe["question"].apply(assign_q_bucket)

    return dataframe


def _compute_pipeline_rows(
    pipeline: str,
    dataframe: pd.DataFrame,
) -> list[dict]:
    """Compute bucket-level F1 rows for one pipeline."""
    rows: list[dict] = []

    dataframe["f1"] = [
        token_f1(row["generated_answer"], row["gold_answer"])
        for _, row in dataframe.iterrows()
    ]

    for bucket, group in dataframe.groupby("q_bucket"):
        f1_mean, lower_bound, upper_bound = bootstrap_ci(
            group["f1"].tolist()
        )

        rows.append(
            {
                "pipeline": pipeline,
                "q_bucket": bucket,
                "n": len(group),
                "f1": f1_mean,
                "f1_lo": lower_bound,
                "f1_hi": upper_bound,
                "indicative": is_indicative(len(group)),
            }
        )

    return rows


def main() -> None:
    """Build and save question-bucket metrics for all pipelines."""
    rows: list[dict] = []

    for pipeline in PIPELINE_KEYS:
        dataframe = _load_answerable_answers(pipeline)

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