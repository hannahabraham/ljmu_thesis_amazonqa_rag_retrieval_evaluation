"""Aggregate generation-quality and lexical faithfulness metrics."""

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
from src.evaluation.faithfulness import groundedness, hallucination_rate_row
from src.evaluation.generation_metrics import (
    bertscore_f1,
    exact_match,
    rouge_l,
    semantic_similarity,
    token_f1,
)
from src.evaluation.statistics import bootstrap_ci
from src.utils.io import parse_list_field
from src.utils.logging_config import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)


def _load_answers(pipeline: str, k_value: int) -> pd.DataFrame:
    """Load generated answers for one pipeline and k value."""
    answers_path = pipeline_output_dir(pipeline) / f"answers_k{k_value}.csv"

    if not answers_path.exists():
        LOGGER.warning("Missing answers file: %s", answers_path)
        return pd.DataFrame()

    dataframe = pd.read_csv(answers_path)

    dataframe["retrieved_context"] = dataframe["retrieved_context"].apply(
        parse_list_field
    )

    return dataframe


def _answerable_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return rows whose gold answer is not unanswerable."""
    return dataframe[
        dataframe["gold_answer"].astype(str).str.upper() != "[UNANSWERABLE]"
    ].reset_index(drop=True)


def _bootstrap_metric(values: list[float]) -> tuple[float, float, float]:
    """Return bootstrap mean and confidence interval for a metric list."""
    return bootstrap_ci(values)


def _compute_answer_metrics(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Compute generation-quality metrics on answerable rows."""
    generated = dataframe["generated_answer"].fillna("").tolist()
    gold = dataframe["gold_answer"].fillna("").tolist()

    exact_matches = [
        exact_match(row["generated_answer"], row["gold_answer"])
        for _, row in dataframe.iterrows()
    ]

    token_f1_scores = [
        token_f1(row["generated_answer"], row["gold_answer"])
        for _, row in dataframe.iterrows()
    ]

    rouge_scores = [
        rouge_l(row["generated_answer"], row["gold_answer"])
        for _, row in dataframe.iterrows()
    ]

    bert_scores = bertscore_f1(generated, gold)
    similarity_scores = semantic_similarity(generated, gold)

    em_mean, em_lower, em_upper = _bootstrap_metric(
        [float(value) for value in exact_matches]
    )
    f1_mean, f1_lower, f1_upper = _bootstrap_metric(token_f1_scores)
    rouge_mean, rouge_lower, rouge_upper = _bootstrap_metric(rouge_scores)
    bert_mean, bert_lower, bert_upper = _bootstrap_metric(bert_scores)
    sim_mean, sim_lower, sim_upper = _bootstrap_metric(similarity_scores)

    return {
        "em": em_mean,
        "em_lo": em_lower,
        "em_hi": em_upper,
        "em_correct": int(sum(exact_matches)),
        "f1": f1_mean,
        "f1_lo": f1_lower,
        "f1_hi": f1_upper,
        "rouge_l": rouge_mean,
        "rouge_l_lo": rouge_lower,
        "rouge_l_hi": rouge_upper,
        "bertscore_f1": bert_mean,
        "bertscore_f1_lo": bert_lower,
        "bertscore_f1_hi": bert_upper,
        "semantic_similarity": sim_mean,
        "semantic_similarity_lo": sim_lower,
        "semantic_similarity_hi": sim_upper,
    }


def _compute_faithfulness_metrics(
    dataframe: pd.DataFrame,
) -> dict[str, float]:
    """Compute lexical groundedness and hallucination over non-refusals."""
    if "refused" in dataframe.columns:
        refused_flags = dataframe["refused"].astype(bool).tolist()
    else:
        refused_flags = [False] * len(dataframe)

    grounded_scores: list[float] = []
    hallucination_scores: list[float] = []

    for (_, row), is_refused in zip(dataframe.iterrows(), refused_flags):
        if is_refused:
            continue

        generated_answer = str(row.get("generated_answer", ""))
        retrieved_context = row["retrieved_context"]

        grounded_score = groundedness(
            generated_answer,
            retrieved_context,
        )
        hallucination_score = hallucination_rate_row(
            generated_answer,
            retrieved_context,
        )

        if grounded_score == grounded_score:
            grounded_scores.append(grounded_score)

        if hallucination_score == hallucination_score:
            hallucination_scores.append(hallucination_score)

    grounded_mean, grounded_lower, grounded_upper = _bootstrap_metric(
        grounded_scores
    )
    halluc_mean, halluc_lower, halluc_upper = _bootstrap_metric(
        hallucination_scores
    )

    return {
        "groundedness": grounded_mean,
        "groundedness_lo": grounded_lower,
        "groundedness_hi": grounded_upper,
        "hallucination_rate": halluc_mean,
        "hallucination_rate_lo": halluc_lower,
        "hallucination_rate_hi": halluc_upper,
    }


def _upsert_pipeline_metrics(
    pipeline: str,
    row: dict[str, Any],
) -> None:
    """Write or update per-pipeline generation metrics."""
    output_path = pipeline_output_dir(pipeline) / "generation_metrics.csv"

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
    """Compute and save generation metrics for all pipelines and k values."""
    rows: list[dict[str, Any]] = []

    for pipeline, k_value in product(PIPELINE_KEYS, K_VALUES):
        full_dataframe = _load_answers(pipeline, k_value)

        if full_dataframe.empty:
            continue

        answerable_dataframe = _answerable_rows(full_dataframe)

        if answerable_dataframe.empty:
            continue

        row = {
            "pipeline": pipeline,
            "k": k_value,
            "n": len(answerable_dataframe),
            **_compute_answer_metrics(answerable_dataframe),
            **_compute_faithfulness_metrics(full_dataframe),
        }

        rows.append(row)

        _upsert_pipeline_metrics(pipeline, row)

    output_path = OUTPUT_DIR / "generation_metrics.csv"

    pd.DataFrame(rows).to_csv(output_path, index=False)

    LOGGER.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()