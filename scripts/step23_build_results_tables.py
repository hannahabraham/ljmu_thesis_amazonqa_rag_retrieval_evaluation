"""Assemble Results Sheet tables from per-question and aggregate outputs."""

from __future__ import annotations

import logging

import pandas as pd

from config.settings import (
    OUTPUT_DIR,
    PER_QUESTION_DIR,
    PIPELINE_KEYS,
    RANDOM_SEED,
    TABLES_DIR,
)
from src.evaluation.table_builders import (
    build_table1_overall,
    build_table2_depth,
    build_table3_category,
    build_table4_length,
    build_table6_answerability,
    build_table7_final_ranking,
)
from src.utils.io import load_per_question
from src.utils.logging_config import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)

RAGAS_COLUMNS = [
    "pipeline",
    "k",
    "faithfulness",
    "context_precision",
    "context_recall",
]

RETRIEVAL_COLUMNS = [
    "pipeline",
    "k",
    "recall_at_k",
    "mrr",
]


def _load_ragas() -> pd.DataFrame:
    """Load aggregate RAGAS metrics, or return an empty placeholder."""
    path = OUTPUT_DIR / "ragas_metrics.csv"

    if not path.exists():
        LOGGER.warning(
            "No RAGAS metrics at %s; Faithfulness/CP/CR will be blank",
            path,
        )
        return pd.DataFrame(columns=RAGAS_COLUMNS)

    return pd.read_csv(path)


def _load_retrieval() -> pd.DataFrame:
    """Load aggregate retrieval metrics, or return an empty placeholder."""
    path = OUTPUT_DIR / "retrieval_metrics.csv"

    if not path.exists():
        LOGGER.warning(
            "No retrieval metrics at %s; Table 2 Recall@K/MRR will be blank",
            path,
        )
        return pd.DataFrame(columns=RETRIEVAL_COLUMNS)

    return pd.read_csv(path)


def _ensure_per_row_ragas(
    per_question: pd.DataFrame,
    ragas: pd.DataFrame,
) -> pd.DataFrame:
    """Fill missing row-level RAGAS columns from aggregate metrics."""
    metric_columns = (
        "faithfulness",
        "context_precision",
        "context_recall",
    )

    for column_name in metric_columns:
        if column_name not in per_question.columns:
            per_question[column_name] = pd.NA

    if ragas.empty:
        return per_question

    aggregate = ragas.rename(
        columns={
            column_name: f"_agg_{column_name}"
            for column_name in metric_columns
        }
    )

    aggregate = aggregate[
        ["pipeline", "k"]
        + [f"_agg_{column_name}" for column_name in metric_columns]
    ]

    merged = per_question.merge(
        aggregate,
        on=["pipeline", "k"],
        how="left",
    )

    for column_name in metric_columns:
        aggregate_column = f"_agg_{column_name}"

        merged[column_name] = (
            pd.to_numeric(
                merged[column_name],
                errors="coerce",
            )
            .fillna(merged[aggregate_column])
        )

    return merged.drop(
        columns=[
            f"_agg_{column_name}"
            for column_name in metric_columns
            if f"_agg_{column_name}" in merged.columns
        ]
    )


def _write_table(
    dataframe: pd.DataFrame,
    filename: str,
) -> None:
    """Write a table CSV to the tables directory and log row count."""
    output_path = TABLES_DIR / filename

    dataframe.to_csv(output_path, index=False)

    LOGGER.info(
        "Wrote %s (%d rows)",
        output_path,
        len(dataframe),
    )


def main() -> None:
    """Build and write all Results Sheet tables."""
    per_question = load_per_question(
        PER_QUESTION_DIR,
        pipelines=list(PIPELINE_KEYS),
        seed=RANDOM_SEED,
    )

    if per_question.empty:
        raise SystemExit(
            f"No per-question JSONL in {PER_QUESTION_DIR}. "
            "Run the pipeline scripts first."
        )

    ragas = _load_ragas()
    retrieval = _load_retrieval()

    per_question = _ensure_per_row_ragas(
        per_question,
        ragas,
    )

    table7 = build_table7_final_ranking(
        per_question,
        ragas_df=ragas,
    )
    _write_table(table7, "table7_final_ranking.csv")

    rank_lookup = dict(
        zip(
            table7["Pipeline"],
            table7["Rank"],
        )
    )

    table1 = build_table1_overall(
        per_question,
        ragas_df=ragas,
    )
    table1["Rank"] = (
        table1["Architecture / Method"]
        .map(rank_lookup)
        .fillna("")
    )
    _write_table(table1, "table1_overall.csv")

    table2 = build_table2_depth(
        per_question,
        retrieval_df=retrieval,
        ragas_df=ragas,
    )
    _write_table(table2, "table2_depth.csv")

    table3 = build_table3_category(
        per_question,
        ragas_df=ragas,
    )
    _write_table(table3, "table3_category.csv")

    table4 = build_table4_length(
        per_question,
        ragas_df=ragas,
    )
    _write_table(table4, "table4_length.csv")

    table6 = build_table6_answerability(
        per_question,
        ragas_df=ragas,
    )
    _write_table(table6, "table6_answerability.csv")


if __name__ == "__main__":
    main()
