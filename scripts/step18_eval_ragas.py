"""Evaluate RAGAS metrics across pipelines and k values.

Computes:
    - faithfulness
    - context_precision
    - context_recall

Per-row scores are written back into the per-question JSONL files so downstream
tables can filter RAGAS metrics by metadata such as category and question length.
"""

from __future__ import annotations

import argparse
import logging
from itertools import product

import pandas as pd

from config.settings import (
    K_VALUES,
    OUTPUT_DIR,
    PER_QUESTION_DIR,
    PIPELINE_KEYS,
    RANDOM_SEED,
    pipeline_output_dir,
)
from src.evaluation.ragas_metrics import run_ragas
from src.utils.io import (
    parse_list_field,
    read_jsonl,
    write_jsonl,
)
from src.utils.logging_config import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)

RAGAS_COLUMNS = (
    "faithfulness",
    "context_precision",
    "context_recall",
)


def _load_per_question(
    pipeline: str,
    k_value: int,
    seed: int,
) -> pd.DataFrame | None:
    """Load per-question rows from JSONL or fallback CSV."""
    jsonl_path = (
        PER_QUESTION_DIR /
        f"{pipeline}_k{k_value}_seed{seed}.jsonl"
    )

    if jsonl_path.exists():
        rows = read_jsonl(jsonl_path)

        if rows:
            return pd.DataFrame(rows)

    csv_path = (
        pipeline_output_dir(pipeline) /
        f"answers_k{k_value}.csv"
    )

    if csv_path.exists():
        dataframe = pd.read_csv(csv_path)

        if "retrieved_context" in dataframe.columns:
            dataframe["retrieved_context"] = (
                dataframe["retrieved_context"]
                .apply(parse_list_field)
            )

        if "retrieved_doc_ids" in dataframe.columns:
            dataframe["retrieved_doc_ids"] = (
                dataframe["retrieved_doc_ids"]
                .apply(parse_list_field)
            )

        return dataframe

    LOGGER.warning(
        (
            "No per-question data for %s k=%d "
            "(looked at %s and %s)"
        ),
        pipeline,
        k_value,
        jsonl_path,
        csv_path,
    )

    return None


def _write_back_per_row_scores(
    pipeline: str,
    k_value: int,
    seed: int,
    scores: pd.DataFrame,
) -> None:
    """Persist row-level RAGAS metrics back into JSONL."""
    jsonl_path = (
        PER_QUESTION_DIR /
        f"{pipeline}_k{k_value}_seed{seed}.jsonl"
    )

    if not jsonl_path.exists():
        LOGGER.info(
            "Skipping JSONL write-back because %s does not exist",
            jsonl_path,
        )
        return

    rows = read_jsonl(jsonl_path)

    if len(rows) != len(scores):
        LOGGER.warning(
            (
                "Length mismatch writing per-row scores "
                "(%d JSONL vs %d RAGAS) for %s k=%d"
            ),
            len(rows),
            len(scores),
            pipeline,
            k_value,
        )
        return

    available_columns = [
        column_name
        for column_name in RAGAS_COLUMNS
        if column_name in scores.columns
    ]

    for row, (_, score_row) in zip(rows, scores.iterrows()):
        for column_name in available_columns:
            value = score_row[column_name]

            row[column_name] = (
                float(value)
                if pd.notna(value)
                else None
            )

    write_jsonl(rows, jsonl_path)

    LOGGER.info(
        "Updated %s with %d per-row RAGAS scores",
        jsonl_path,
        len(rows),
    )


def _prepare_ragas_input(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Convert per-question rows into RAGAS input format."""
    if "retrieved_context" in dataframe.columns:
        contexts = dataframe["retrieved_context"].apply(
            lambda value: (
                value
                if isinstance(value, list)
                else parse_list_field(value)
            )
        )
    else:
        contexts = pd.Series(
            [[] for _ in range(len(dataframe))]
        )

    return pd.DataFrame(
        {
            "question": dataframe["question"],
            "answer": dataframe["generated_answer"].fillna(""),
            "contexts": contexts,
            "ground_truth": dataframe["gold_answer"],
        }
    )


def _aggregate_scores(
    scores: pd.DataFrame,
    pipeline: str,
    k_value: int,
    n_rows: int,
) -> dict:
    """Compute aggregate mean RAGAS metrics."""
    row = {
        "pipeline": pipeline,
        "k": k_value,
        "n": n_rows,
    }

    for column_name in RAGAS_COLUMNS:
        if column_name in scores.columns:
            row[column_name] = float(
                scores[column_name].mean()
            )
        else:
            row[column_name] = float("nan")

    return row


def _upsert_pipeline_metrics(
    pipeline: str,
    row: dict,
) -> None:
    """Write or update per-pipeline RAGAS metrics."""
    output_path = (
        pipeline_output_dir(pipeline) /
        "ragas_metrics.csv"
    )

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

    merged = (
        pd.concat(
            [existing, pd.DataFrame([row])],
            ignore_index=True,
        )
        .sort_values("k")
        .reset_index(drop=True)
    )

    merged.to_csv(output_path, index=False)


def _evaluate(
    pipeline: str,
    k_value: int,
    seed: int,
) -> dict | None:
    """Run RAGAS evaluation for one pipeline/k combination."""
    dataframe = _load_per_question(
        pipeline,
        k_value,
        seed,
    )

    if dataframe is None or dataframe.empty:
        return None

    ragas_input = _prepare_ragas_input(dataframe)

    result = run_ragas(ragas_input)

    scores = (
        result.to_pandas()
        if hasattr(result, "to_pandas")
        else pd.DataFrame(result)
    )

    aggregate_row = _aggregate_scores(
        scores,
        pipeline,
        k_value,
        len(ragas_input),
    )

    _write_back_per_row_scores(
        pipeline,
        k_value,
        seed,
        scores,
    )

    _upsert_pipeline_metrics(
        pipeline,
        aggregate_row,
    )

    return aggregate_row


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=None,
        help=(
            "k values to evaluate "
            "(default: all K_VALUES)"
        ),
    )

    parser.add_argument(
        "--k5-only",
        action="store_true",
        help=(
            "Evaluate only k=5 "
            "(ignored when --ks is provided)"
        ),
    )

    parser.add_argument(
        "--pipelines",
        nargs="+",
        default=None,
        choices=list(PIPELINE_KEYS),
        help="Pipelines to evaluate (default: all)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
    )

    return parser


def main() -> None:
    """Run RAGAS evaluation and save aggregate outputs."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.ks:
        k_values = args.ks
    elif args.k5_only:
        k_values = [5]
    else:
        k_values = list(K_VALUES)

    pipelines = args.pipelines or list(PIPELINE_KEYS)

    combinations = list(product(pipelines, k_values))

    LOGGER.info(
        "Evaluating %d (pipeline, k) cells: %s",
        len(combinations),
        combinations,
    )

    rows: list[dict] = []

    for pipeline, k_value in combinations:
        result = _evaluate(
            pipeline,
            k_value,
            seed=args.seed,
        )

        if result is not None:
            rows.append(result)

    output_path = OUTPUT_DIR / "ragas_metrics.csv"

    new_rows = pd.DataFrame(rows)

    if output_path.exists() and not new_rows.empty:
        existing = pd.read_csv(output_path)

        keys = set(
            zip(
                new_rows["pipeline"],
                new_rows["k"],
            )
        )

        existing = existing[
            ~existing.apply(
                lambda row: (
                    row["pipeline"],
                    row["k"],
                ) in keys,
                axis=1,
            )
        ]

        merged = pd.concat(
            [existing, new_rows],
            ignore_index=True,
        )
    else:
        merged = new_rows

    merged = (
        merged
        .sort_values(["pipeline", "k"])
        .reset_index(drop=True)
    )

    merged.to_csv(output_path, index=False)

    LOGGER.info(
        "Wrote %s (%d rows)",
        output_path,
        len(merged),
    )


if __name__ == "__main__":
    main()