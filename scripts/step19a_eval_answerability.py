"""Compute answerability, long-context, and noise-robustness metrics."""

from __future__ import annotations

import logging
from itertools import product

import pandas as pd

from config.settings import (
    K_VALUES,
    OUTPUT_DIR,
    PIPELINE_KEYS,
    pipeline_output_dir,
)
from src.evaluation.answerability import compute_answerability_table
from src.evaluation.robustness import (
    long_context_metrics,
    noise_robustness_metrics,
)
from src.evaluation.statistics import wilson_ci
from src.utils.io import parse_list_field
from src.utils.logging_config import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)


def _load_answers(pipeline: str, k_value: int) -> pd.DataFrame:
    """Load answers for one pipeline and k value."""
    answers_path = pipeline_output_dir(pipeline) / f"answers_k{k_value}.csv"

    if not answers_path.exists():
        LOGGER.warning("Missing answers file: %s", answers_path)
        return pd.DataFrame()

    dataframe = pd.read_csv(answers_path)

    dataframe["retrieved_doc_ids"] = dataframe["retrieved_doc_ids"].apply(
        parse_list_field
    )
    dataframe["retrieved_context"] = dataframe["retrieved_context"].apply(
        parse_list_field
    )

    return dataframe


def _compute_answerability_metrics(dataframe: pd.DataFrame) -> dict | None:
    """Compute answerability metrics and Wilson confidence interval."""
    labelled = dataframe[dataframe["is_answerable"].notna()].copy()

    if labelled.empty:
        return None

    labelled["is_answerable"] = labelled["is_answerable"].astype(int)

    table = compute_answerability_table(labelled)

    total = int(table["n"].iloc[0])
    correctly_answered = int(table["correctly_answered"].iloc[0])
    correctly_refused = int(table["correctly_refused"].iloc[0])
    wrongly_refused = int(table["wrongly_refused"].iloc[0])
    wrongly_answered = int(table["wrongly_answered"].iloc[0])

    successes = correctly_answered + correctly_refused

    accuracy, lower_bound, upper_bound = wilson_ci(
        successes,
        total,
    )

    return {
        "n": total,
        "correctly_answered": correctly_answered,
        "wrongly_refused": wrongly_refused,
        "correctly_refused": correctly_refused,
        "wrongly_answered": wrongly_answered,
        "answerability_acc": accuracy,
        "answerability_acc_lo": lower_bound,
        "answerability_acc_hi": upper_bound,
    }


def main() -> None:
    """Compute and save answerability metrics for all pipelines and k values."""
    rows: list[dict] = []

    for pipeline, k_value in product(PIPELINE_KEYS, K_VALUES):
        full_dataframe = _load_answers(pipeline, k_value)

        if full_dataframe.empty:
            continue

        answerability = _compute_answerability_metrics(full_dataframe)

        if answerability is None:
            continue

        rows.append({
            "pipeline": pipeline,
            "k": k_value,
            **answerability,
            **long_context_metrics(full_dataframe),
            **noise_robustness_metrics(full_dataframe, k_value),
        })

    output_path = OUTPUT_DIR / "answerability_metrics.csv"

    pd.DataFrame(rows).to_csv(output_path, index=False)

    LOGGER.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
