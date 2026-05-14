"""Aggregate retrieval metrics across all pipeline and k-value results."""

from __future__ import annotations

import logging
from itertools import product
from typing import Any

import pandas as pd

from config.settings import (
    K_VALUES,
    OUTPUT_DIR,
    PIPELINE_KEYS,
    pipeline_output_dir,
)
from src.evaluation.retrieval_metrics import (
    hit_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.evaluation.statistics import bootstrap_ci
from src.utils.io import parse_list_field
from src.utils.logging_config import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)


def _load_retrieval_rows(pipeline: str, k_value: int) -> pd.DataFrame:
    """Load rows with available golden evidence for one pipeline/k value."""
    answers_path = pipeline_output_dir(pipeline) / f"answers_k{k_value}.csv"

    if not answers_path.exists():
        LOGGER.warning("Missing %s, skipping", answers_path)
        return pd.DataFrame()

    dataframe = pd.read_csv(answers_path)
    dataframe = dataframe[dataframe["evidence_doc_id"].notna()].copy()

    if dataframe.empty:
        return dataframe

    dataframe["retrieved_doc_ids"] = dataframe["retrieved_doc_ids"].apply(
        parse_list_field
    )

    return dataframe


def _metric_ci(values: list[float]) -> tuple[float, float, float]:
    """Return bootstrap mean and confidence interval for a metric."""
    return bootstrap_ci(values)


def _compute_metric_row(
    dataframe: pd.DataFrame,
    pipeline: str,
    k_value: int,
) -> dict[str, Any]:
    """Compute retrieval metrics for one pipeline/k value."""
    hits = [
        hit_at_k(row["retrieved_doc_ids"], row["evidence_doc_id"], k_value)
        for _, row in dataframe.iterrows()
    ]

    recalls = [
        recall_at_k(row["retrieved_doc_ids"], row["evidence_doc_id"], k_value)
        for _, row in dataframe.iterrows()
    ]

    reciprocal_ranks = [
        reciprocal_rank(row["retrieved_doc_ids"], row["evidence_doc_id"])
        for _, row in dataframe.iterrows()
    ]

    ndcgs = [
        ndcg_at_k(row["retrieved_doc_ids"], row["evidence_doc_id"], k_value)
        for _, row in dataframe.iterrows()
    ]

    hit_mean, hit_lower, hit_upper = _metric_ci(
        [float(value) for value in hits]
    )
    recall_mean, recall_lower, recall_upper = _metric_ci(
        [float(value) for value in recalls]
    )
    mrr_mean, mrr_lower, mrr_upper = _metric_ci(
        [float(value) for value in reciprocal_ranks]
    )
    ndcg_mean, ndcg_lower, ndcg_upper = _metric_ci(
        [float(value) for value in ndcgs]
    )

    return {
        "pipeline": pipeline,
        "k": k_value,
        "n": len(dataframe),
        "hit_at_k": hit_mean,
        "hit_at_k_lo": hit_lower,
        "hit_at_k_hi": hit_upper,
        "recall_at_k": recall_mean,
        "recall_at_k_lo": recall_lower,
        "recall_at_k_hi": recall_upper,
        "mrr": mrr_mean,
        "mrr_lo": mrr_lower,
        "mrr_hi": mrr_upper,
        "ndcg_at_k": ndcg_mean,
        "ndcg_at_k_lo": ndcg_lower,
        "ndcg_at_k_hi": ndcg_upper,
    }


def _upsert_pipeline_metrics(
    pipeline: str,
    row: dict[str, Any],
) -> None:
    """Write or update the per-pipeline retrieval metrics CSV."""
    output_path = pipeline_output_dir(pipeline) / "retrieval_metrics.csv"

    if output_path.exists():
        existing = pd.read_csv(output_path)
    else:
        existing = pd.DataFrame()

    if not existing.empty:
        existing = existing[
            ~(
                (existing["pipeline"] == row["pipeline"])
                & (existing["k"] == row["k"])
            )
        ]

    output = (
        pd.concat(
            [existing, pd.DataFrame([row])],
            ignore_index=True,
        )
        .sort_values("k")
        .reset_index(drop=True)
    )

    output.to_csv(output_path, index=False)


def main() -> None:
    """Compute and save retrieval metrics for all pipelines and k values."""
    rows: list[dict[str, Any]] = []

    for pipeline, k_value in product(PIPELINE_KEYS, K_VALUES):
        dataframe = _load_retrieval_rows(pipeline, k_value)

        if dataframe.empty:
            continue

        row = _compute_metric_row(
            dataframe,
            pipeline,
            k_value,
        )

        rows.append(row)

        _upsert_pipeline_metrics(
            pipeline,
            row,
        )

    output_path = OUTPUT_DIR / "retrieval_metrics.csv"

    pd.DataFrame(rows).to_csv(output_path, index=False)

    LOGGER.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()