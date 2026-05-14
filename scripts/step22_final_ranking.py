"""Generate composite rankings, Wilcoxon comparisons, and sensitivity sweeps.

Outputs:
    outputs/tables/table7_final_ranking.csv
    outputs/tables/pairwise_wilcoxon.csv
    outputs/correct_threshold_sensitivity.csv
    outputs/composite_weight_sensitivity.csv
"""

from __future__ import annotations

import logging

import pandas as pd

from config.settings import (
    CORRECT_F1_SENSITIVITY,
    OUTPUT_DIR,
    PER_QUESTION_DIR,
    PIPELINE_KEYS,
    RANDOM_SEED,
    TABLES_DIR,
)
from src.evaluation.generation_metrics import correct_answers_count
from src.evaluation.table_builders import (
    COMPOSITE_WEIGHTS,
    PIPELINE_LABEL,
    build_pairwise_wilcoxon,
    build_table7_final_ranking,
)
from src.utils.io import load_per_question
from src.utils.logging_config import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)

ALTERNATE_WEIGHT_VECTORS: dict[str, dict[str, float]] = {
    "primary": COMPOSITE_WEIGHTS,
    "f1_heavy": {
        "f1": 0.40,
        "faithfulness": 0.20,
        "context_precision": 0.10,
        "context_recall": 0.05,
        "answerability_acc": 0.15,
        "category_consistency": 0.05,
        "latency": 0.05,
    },
    "faithfulness_heavy": {
        "f1": 0.15,
        "faithfulness": 0.35,
        "context_precision": 0.15,
        "context_recall": 0.10,
        "answerability_acc": 0.15,
        "category_consistency": 0.05,
        "latency": 0.05,
    },
    "balanced_quality": {
        "f1": 0.25,
        "faithfulness": 0.25,
        "context_precision": 0.15,
        "context_recall": 0.15,
        "answerability_acc": 0.15,
        "category_consistency": 0.00,
        "latency": 0.05,
    },
}


def _load_full_per_question() -> pd.DataFrame:
    """Load all per-question pipeline outputs."""
    dataframe = load_per_question(
        PER_QUESTION_DIR,
        pipelines=list(PIPELINE_KEYS),
        seed=RANDOM_SEED,
    )

    if dataframe.empty:
        raise SystemExit(
            f"No per-question JSONL found in {PER_QUESTION_DIR}"
        )

    return dataframe


def _attach_ragas(per_question: pd.DataFrame) -> pd.DataFrame:
    """Attach aggregate RAGAS metrics to per-question rows."""
    ragas_path = OUTPUT_DIR / "ragas_metrics.csv"

    if not ragas_path.exists():
        LOGGER.warning(
            "No RAGAS metrics found at %s",
            ragas_path,
        )
        return per_question

    ragas = pd.read_csv(ragas_path)

    renamed = ragas.rename(
        columns={
            "faithfulness": "_agg_faithfulness",
            "context_precision": "_agg_context_precision",
            "context_recall": "_agg_context_recall",
        }
    )

    ragas_columns = [
        "pipeline",
        "k",
        "_agg_faithfulness",
        "_agg_context_precision",
        "_agg_context_recall",
    ]

    return per_question.merge(
        renamed[ragas_columns],
        on=["pipeline", "k"],
        how="left",
    )


def _fill_missing_ragas(per_question: pd.DataFrame) -> pd.DataFrame:
    """Broadcast aggregate RAGAS metrics into missing row-level columns."""
    metric_columns = (
        "faithfulness",
        "context_precision",
        "context_recall",
    )

    for column_name in metric_columns:
        if column_name not in per_question.columns:
            per_question[column_name] = pd.NA

        aggregate_column = f"_agg_{column_name}"

        if aggregate_column in per_question.columns:
            per_question[column_name] = (
                pd.to_numeric(
                    per_question[column_name],
                    errors="coerce",
                )
                .fillna(per_question[aggregate_column])
            )

    return per_question


def _write_correct_threshold_sensitivity(
    per_question: pd.DataFrame,
) -> None:
    """Recompute Correct Answers for multiple F1 thresholds."""
    k5_dataframe = per_question[per_question["k"] == 5]

    rows: list[dict] = []

    for pipeline in PIPELINE_KEYS:
        subset = k5_dataframe[k5_dataframe["pipeline"] == pipeline]

        if subset.empty:
            continue

        for threshold in CORRECT_F1_SENSITIVITY:
            rows.append(
                {
                    "pipeline": pipeline,
                    "pipeline_label": PIPELINE_LABEL.get(
                        pipeline,
                        pipeline,
                    ),
                    "f1_threshold": threshold,
                    "correct_answers": correct_answers_count(
                        subset,
                        threshold,
                    ),
                    "total_questions": int(len(subset)),
                }
            )

    dataframe = pd.DataFrame(rows)

    output_path = OUTPUT_DIR / "correct_threshold_sensitivity.csv"

    dataframe.to_csv(output_path, index=False)

    LOGGER.info("Wrote %s", output_path)


def _write_weight_sensitivity(
    per_question: pd.DataFrame,
) -> None:
    """Evaluate ranking stability under alternate weight vectors."""
    rows: list[dict] = []

    for weight_name, weights in ALTERNATE_WEIGHT_VECTORS.items():
        ranking = build_table7_final_ranking(
            per_question,
            weights=weights,
        )

        for _, row in ranking.iterrows():
            rows.append(
                {
                    "weight_vector": weight_name,
                    "pipeline": row["Pipeline"],
                    "best_k": row["Best K"],
                    "composite_score": row["Composite Score"],
                    "rank": row["Rank"],
                }
            )

    dataframe = pd.DataFrame(rows)

    output_path = OUTPUT_DIR / "composite_weight_sensitivity.csv"

    dataframe.to_csv(output_path, index=False)

    LOGGER.info("Wrote %s", output_path)

    primary_top = (
        dataframe[dataframe["weight_vector"] == "primary"]
        .sort_values("rank")["pipeline"]
        .iloc[0]
    )

    for weight_name in ALTERNATE_WEIGHT_VECTORS:
        if weight_name == "primary":
            continue

        alternate_top = (
            dataframe[dataframe["weight_vector"] == weight_name]
            .sort_values("rank")["pipeline"]
            .iloc[0]
        )

        if alternate_top != primary_top:
            LOGGER.warning(
                (
                    "Ranking changes under %s "
                    "(top=%s, primary=%s)"
                ),
                weight_name,
                alternate_top,
                primary_top,
            )


def main() -> None:
    """Build ranking outputs and statistical sensitivity reports."""
    per_question = _load_full_per_question()

    per_question = _attach_ragas(per_question)

    per_question = _fill_missing_ragas(per_question)

    table7 = build_table7_final_ranking(per_question)

    table7_path = TABLES_DIR / "table7_final_ranking.csv"

    table7.to_csv(table7_path, index=False)

    LOGGER.info(
        "Wrote %s (%d rows)",
        table7_path,
        len(table7),
    )

    pairwise = build_pairwise_wilcoxon(
        per_question,
        metrics=("token_f1", "faithfulness"),
    )

    pairwise_path = TABLES_DIR / "pairwise_wilcoxon.csv"

    pairwise.to_csv(pairwise_path, index=False)

    LOGGER.info(
        "Wrote %s (%d rows)",
        pairwise_path,
        len(pairwise),
    )

    _write_correct_threshold_sensitivity(per_question)

    _write_weight_sensitivity(per_question)


if __name__ == "__main__":
    main()