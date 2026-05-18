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
    rouge_l,
    semantic_similarity,
    token_f1,
    yesno_em,
)
from src.evaluation.latency import latency_detail
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
    """Compute generation-quality metrics on answerable rows.

    Token-F1 / ROUGE-L / Semantic Similarity span all answerable rows.
    Yes/No EM is computed on the yes/no slice only (gold answer starts with
    "yes" or "no" after normalisation).
    """
    generated = dataframe["generated_answer"].fillna("").tolist()
    gold = dataframe["gold_answer"].fillna("").tolist()

    token_f1_scores = [
        token_f1(row["generated_answer"], row["gold_answer"])
        for _, row in dataframe.iterrows()
    ]

    rouge_scores = [
        rouge_l(row["generated_answer"], row["gold_answer"])
        for _, row in dataframe.iterrows()
    ]

    similarity_scores = semantic_similarity(generated, gold)

    f1_mean, f1_lower, f1_upper = _bootstrap_metric(token_f1_scores)
    rouge_mean, rouge_lower, rouge_upper = _bootstrap_metric(rouge_scores)
    sim_mean, sim_lower, sim_upper = _bootstrap_metric(similarity_scores)

    yesno_mask = dataframe["gold_answer"].astype(str).str.strip().str.lower().str.match(
        r"^(yes|no)\b"
    )
    yesno_subset = dataframe[yesno_mask.fillna(False)]
    yesno_scores = [
        float(yesno_em(row["generated_answer"], row["gold_answer"]))
        for _, row in yesno_subset.iterrows()
    ]
    if yesno_scores:
        yesno_mean, yesno_lower, yesno_upper = _bootstrap_metric(yesno_scores)
        yesno_correct = int(sum(yesno_scores))
    else:
        yesno_mean = yesno_lower = yesno_upper = float("nan")
        yesno_correct = 0

    return {
        "yesno_em": yesno_mean,
        "yesno_em_lo": yesno_lower,
        "yesno_em_hi": yesno_upper,
        "yesno_em_n": int(len(yesno_subset)),
        "yesno_em_correct": yesno_correct,
        "f1": f1_mean,
        "f1_lo": f1_lower,
        "f1_hi": f1_upper,
        "rouge_l": rouge_mean,
        "rouge_l_lo": rouge_lower,
        "rouge_l_hi": rouge_upper,
        "semantic_similarity": sim_mean,
        "semantic_similarity_lo": sim_lower,
        "semantic_similarity_hi": sim_upper,
    }


def _compute_latency_metrics(dataframe: pd.DataFrame) -> dict[str, float]:
    """Mean + p50/p95 latency for retrieval, generation, and total."""
    detail = latency_detail(
        dataframe.assign(
            total_ms=dataframe["retrieval_ms"].astype(float)
            + dataframe["generation_ms"].astype(float)
        )
    )
    retrieval_mean = float(dataframe["retrieval_ms"].astype(float).mean())
    generation_mean = float(dataframe["generation_ms"].astype(float).mean())
    return {
        "retrieval_ms_mean": retrieval_mean,
        "generation_ms_mean": generation_mean,
        **detail,
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


def main() -> None:
    """Compute and save generation metrics for all pipelines and k values."""
    rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []

    for pipeline, k_value in product(PIPELINE_KEYS, K_VALUES):
        full_dataframe = _load_answers(pipeline, k_value)

        if full_dataframe.empty:
            continue

        answerable_dataframe = _answerable_rows(full_dataframe)

        if answerable_dataframe.empty:
            continue

        latency = _compute_latency_metrics(full_dataframe)
        rows.append({
            "pipeline": pipeline,
            "k": k_value,
            "n": len(answerable_dataframe),
            **_compute_answer_metrics(answerable_dataframe),
            **_compute_faithfulness_metrics(full_dataframe),
        })

        latency_rows.append(
            {"pipeline": pipeline, "k": k_value, "n": len(full_dataframe), **latency}
        )

    output_path = OUTPUT_DIR / "generation_metrics.csv"

    pd.DataFrame(rows).to_csv(output_path, index=False)

    LOGGER.info("Wrote %s", output_path)

    latency_path = OUTPUT_DIR / "latency_detail.csv"
    pd.DataFrame(latency_rows).to_csv(latency_path, index=False)
    LOGGER.info("Wrote %s", latency_path)


if __name__ == "__main__":
    main()
