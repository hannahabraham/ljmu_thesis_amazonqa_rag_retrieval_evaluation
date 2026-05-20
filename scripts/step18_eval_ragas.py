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
import json
import logging
import random
import time
from itertools import product
from typing import Any

import pandas as pd

from config.settings import (
    GEMINI_JUDGE_MODEL,
    K_VALUES,
    OUTPUT_DIR,
    PER_QUESTION_DIR,
    PIPELINE_KEYS,
    RAGAS_BACKOFF_MULTIPLIER,
    RAGAS_BACKOFF_SECONDS,
    RAGAS_BATCH_SIZE,
    RAGAS_MAX_RETRIES,
    RAGAS_SLEEP_BETWEEN_BATCHES,
    RANDOM_SEED,
    pipeline_output_dir,
)
from src.evaluation.ragas_metrics import run_ragas
from src.llm_clients.error_terms import is_daily_quota_error, should_try_next_key
from src.utils.caching import get_cached, set_cached
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

RAGAS_CACHE_NAMESPACE = "ragas_row_v1"


class DailyQuotaReached(RuntimeError):
    """Raised when the Gemini project appears to have hit daily quota."""


def _load_per_question(
    pipeline: str,
    k_value: int,
    seed: int,
) -> pd.DataFrame | None:
    """Load per-question rows from JSONL or fallback CSV."""
    jsonl_path = PER_QUESTION_DIR / f"{pipeline}_k{k_value}_seed{seed}.jsonl"

    if jsonl_path.exists():
        rows = read_jsonl(jsonl_path)

        if rows:
            return pd.DataFrame(rows)

    csv_path = pipeline_output_dir(pipeline) / f"answers_k{k_value}.csv"

    if csv_path.exists():
        dataframe = pd.read_csv(csv_path)

        if "retrieved_context" in dataframe.columns:
            dataframe["retrieved_context"] = dataframe["retrieved_context"].apply(
                parse_list_field
            )

        if "retrieved_doc_ids" in dataframe.columns:
            dataframe["retrieved_doc_ids"] = dataframe["retrieved_doc_ids"].apply(
                parse_list_field
            )

        return dataframe

    LOGGER.warning(
        ("No per-question data for %s k=%d " "(looked at %s and %s)"),
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
    jsonl_path = PER_QUESTION_DIR / f"{pipeline}_k{k_value}_seed{seed}.jsonl"

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
        column_name for column_name in RAGAS_COLUMNS if column_name in scores.columns
    ]

    for row, (_, score_row) in zip(rows, scores.iterrows()):
        for column_name in available_columns:
            value = score_row[column_name]

            row[column_name] = float(value) if pd.notna(value) else None

        row["ragas_attempted"] = True

    write_jsonl(rows, jsonl_path)

    LOGGER.info(
        "Updated %s with %d per-row RAGAS scores",
        jsonl_path,
        len(rows),
    )


def _write_back_indexed_scores(
    pipeline: str,
    k_value: int,
    seed: int,
    indexed_scores: dict[int, dict[str, float | None]],
) -> None:
    """Persist a partial set of row-level RAGAS scores into JSONL."""
    if not indexed_scores:
        return

    jsonl_path = PER_QUESTION_DIR / f"{pipeline}_k{k_value}_seed{seed}.jsonl"

    if not jsonl_path.exists():
        LOGGER.info(
            "Skipping partial JSONL write-back because %s does not exist",
            jsonl_path,
        )
        return

    rows = read_jsonl(jsonl_path)

    for row_index, score_values in indexed_scores.items():
        if row_index < 0 or row_index >= len(rows):
            LOGGER.warning(
                "Skipping out-of-range RAGAS row index %d for %s k=%d",
                row_index,
                pipeline,
                k_value,
            )
            continue

        for column_name in RAGAS_COLUMNS:
            if column_name not in score_values:
                continue

            value = score_values[column_name]
            rows[row_index][column_name] = (
                float(value) if pd.notna(value) else None
            )

        rows[row_index]["ragas_attempted"] = True

    write_jsonl(rows, jsonl_path)

    LOGGER.info(
        "Checkpointed %d RAGAS rows into %s",
        len(indexed_scores),
        jsonl_path,
    )


def _prepare_ragas_input(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Convert per-question rows into RAGAS input format."""
    if "retrieved_context" in dataframe.columns:
        contexts = dataframe["retrieved_context"].apply(
            lambda value: (
                value if isinstance(value, list) else parse_list_field(value)
            )
        )
    else:
        contexts = pd.Series([[] for _ in range(len(dataframe))])

    return pd.DataFrame(
        {
            "question": dataframe["question"],
            "answer": dataframe["generated_answer"].fillna(""),
            "contexts": contexts,
            "ground_truth": dataframe["gold_answer"],
        }
    )


def _normalise_contexts(value: Any) -> list[str]:
    """Return contexts as a stable list of strings for hashing/evaluation."""
    if isinstance(value, list):
        return [str(item) for item in value]

    parsed = parse_list_field(value)
    if isinstance(parsed, list):
        return [str(item) for item in parsed]

    return []


def _cache_parts(row: pd.Series) -> tuple[str, ...]:
    """Build stable cache parts for one RAGAS input row."""
    contexts_json = json.dumps(
        _normalise_contexts(row.get("contexts", [])),
        ensure_ascii=False,
        sort_keys=True,
    )

    return (
        GEMINI_JUDGE_MODEL,
        str(row.get("question", "")),
        str(row.get("answer", "")),
        contexts_json,
        str(row.get("ground_truth", "")),
    )


def _score_is_complete(row: pd.Series) -> bool:
    """Return True when all row-level RAGAS scores are present."""
    attempted = row.get("ragas_attempted", False)

    if attempted is True or str(attempted).lower() == "true":
        return True

    return all(
        column_name in row.index
        and pd.notna(pd.to_numeric(row[column_name], errors="coerce"))
        for column_name in RAGAS_COLUMNS
    )


def _coerce_score_dict(score_row: pd.Series | dict[str, Any]) -> dict[str, float | None]:
    """Convert RAGAS score values to JSON/cache-friendly floats."""
    scores: dict[str, float | None] = {}

    for column_name in RAGAS_COLUMNS:
        if column_name not in score_row:
            continue

        value = score_row[column_name]
        scores[column_name] = float(value) if pd.notna(value) else None

    return scores


def _apply_cached_scores(
    dataframe: pd.DataFrame,
    ragas_input: pd.DataFrame,
    pipeline: str,
    k_value: int,
    seed: int,
) -> list[int]:
    """Fill missing RAGAS scores from disk cache and return still-missing rows."""
    cached_updates: dict[int, dict[str, float | None]] = {}
    pending_indices: list[int] = []

    for row_index, row in dataframe.iterrows():
        if _score_is_complete(row):
            continue

        cached = get_cached(
            RAGAS_CACHE_NAMESPACE,
            *_cache_parts(ragas_input.loc[row_index]),
        )

        if isinstance(cached, dict) and all(
            column_name in cached for column_name in RAGAS_COLUMNS
        ):
            scores = _coerce_score_dict(cached)
            cached_updates[int(row_index)] = scores

            for column_name, value in scores.items():
                dataframe.at[row_index, column_name] = value

            dataframe.at[row_index, "ragas_attempted"] = True

            continue

        pending_indices.append(int(row_index))

    if cached_updates:
        LOGGER.info(
            "Restored %d %s k=%d RAGAS rows from cache",
            len(cached_updates),
            pipeline,
            k_value,
        )
        _write_back_indexed_scores(
            pipeline,
            k_value,
            seed,
            cached_updates,
        )

    return pending_indices


def _run_ragas_batch_with_backoff(
    batch: pd.DataFrame,
    workers: int | None,
    max_retries: int,
    backoff_seconds: float,
    backoff_multiplier: float,
) -> pd.DataFrame:
    """Run one RAGAS batch with quota-friendly exponential backoff."""
    delay = max(0.0, backoff_seconds)

    for attempt in range(max(1, max_retries) + 1):
        try:
            result = run_ragas(batch, workers=workers)
            return (
                result.to_pandas()
                if hasattr(result, "to_pandas")
                else pd.DataFrame(result)
            )

        except Exception as error:  # noqa: BLE001 -- RAGAS bubbles up varied SDK errors
            if is_daily_quota_error(error):
                raise DailyQuotaReached(
                    (
                        "Gemini daily request quota appears to be exhausted. "
                        "Checkpointed rows are preserved; rerun after the RPD reset."
                    )
                ) from error

            if attempt >= max_retries or not should_try_next_key(error):
                raise

            jittered_delay = delay * random.uniform(0.8, 1.3)
            LOGGER.warning(
                (
                    "RAGAS batch failed with a retryable error "
                    "(attempt %d/%d). Sleeping %.1fs. Error: %s"
                ),
                attempt + 1,
                max_retries,
                jittered_delay,
                error,
            )
            time.sleep(jittered_delay)
            delay = max(delay * max(1.0, backoff_multiplier), delay + 1.0)

    raise RuntimeError("RAGAS batch retry loop exited unexpectedly.")


def _iter_batches(indices: list[int], batch_size: int) -> list[list[int]]:
    """Split row indices into fixed-size batches."""
    safe_batch_size = max(1, batch_size)
    return [
        indices[start : start + safe_batch_size]
        for start in range(0, len(indices), safe_batch_size)
    ]


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
            numeric_scores = pd.to_numeric(
                scores[column_name],
                errors="coerce",
            )
            row[column_name] = float(numeric_scores.mean())
        else:
            row[column_name] = float("nan")

    return row


def _evaluate(
    pipeline: str,
    k_value: int,
    seed: int,
    workers: int | None,
    batch_size: int,
    sleep_seconds: float,
    max_retries: int,
    backoff_seconds: float,
    backoff_multiplier: float,
) -> dict | None:
    """Run RAGAS evaluation for one pipeline/k combination."""
    dataframe = _load_per_question(
        pipeline,
        k_value,
        seed,
    )

    if dataframe is None or dataframe.empty:
        return None

    dataframe = dataframe.reset_index(drop=True)

    for column_name in RAGAS_COLUMNS:
        if column_name not in dataframe.columns:
            dataframe[column_name] = pd.NA

    LOGGER.info(
        "Running RAGAS for %s k=%d on %d rows",
        pipeline,
        k_value,
        len(dataframe),
    )

    ragas_input = _prepare_ragas_input(dataframe)

    pending_indices = _apply_cached_scores(
        dataframe,
        ragas_input,
        pipeline,
        k_value,
        seed,
    )

    if pending_indices:
        LOGGER.info(
            (
                "Evaluating %d missing RAGAS rows for %s k=%d "
                "in batches of %d"
            ),
            len(pending_indices),
            pipeline,
            k_value,
            max(1, batch_size),
        )

    batches = _iter_batches(pending_indices, batch_size)

    for batch_number, batch_indices in enumerate(batches, start=1):
        LOGGER.info(
            "RAGAS batch %d for %s k=%d: rows %s",
            batch_number,
            pipeline,
            k_value,
            batch_indices,
        )

        batch_input = ragas_input.loc[batch_indices].reset_index(drop=True)
        batch_scores = _run_ragas_batch_with_backoff(
            batch_input,
            workers=workers,
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
            backoff_multiplier=backoff_multiplier,
        )

        indexed_scores: dict[int, dict[str, float | None]] = {}

        for row_index, (_, score_row) in zip(
            batch_indices,
            batch_scores.iterrows(),
        ):
            scores = _coerce_score_dict(score_row)
            indexed_scores[row_index] = scores

            for column_name, value in scores.items():
                dataframe.at[row_index, column_name] = value

            dataframe.at[row_index, "ragas_attempted"] = True

            set_cached(
                RAGAS_CACHE_NAMESPACE,
                scores,
                *_cache_parts(ragas_input.loc[row_index]),
            )

        _write_back_indexed_scores(
            pipeline,
            k_value,
            seed,
            indexed_scores,
        )

        if sleep_seconds > 0 and batch_number < len(batches):
            LOGGER.info(
                "Sleeping %.1fs before the next RAGAS batch",
                sleep_seconds,
            )
            time.sleep(sleep_seconds)

    scores = dataframe.loc[:, RAGAS_COLUMNS].copy()

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

    return aggregate_row


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=None,
        help=("k values to evaluate " "(default: all K_VALUES)"),
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

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="RAGAS parallel worker count (default: 1)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=RAGAS_BATCH_SIZE,
        help=(
            "Rows per checkpointed RAGAS batch "
            f"(default: {RAGAS_BATCH_SIZE})"
        ),
    )

    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=RAGAS_SLEEP_BETWEEN_BATCHES,
        help=(
            "Pause between RAGAS batches "
            f"(default: {RAGAS_SLEEP_BETWEEN_BATCHES})"
        ),
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=RAGAS_MAX_RETRIES,
        help=f"Retryable failures per RAGAS batch (default: {RAGAS_MAX_RETRIES})",
    )

    parser.add_argument(
        "--backoff-seconds",
        type=float,
        default=RAGAS_BACKOFF_SECONDS,
        help=(
            "Initial retry backoff in seconds "
            f"(default: {RAGAS_BACKOFF_SECONDS})"
        ),
    )

    parser.add_argument(
        "--backoff-multiplier",
        type=float,
        default=RAGAS_BACKOFF_MULTIPLIER,
        help=(
            "Retry backoff multiplier "
            f"(default: {RAGAS_BACKOFF_MULTIPLIER})"
        ),
    )

    return parser


def main() -> None:
    """Run RAGAS evaluation and save aggregate outputs."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.ks:
        k_values = args.ks
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
    stopped_for_daily_quota = False

    for pipeline, k_value in combinations:
        try:
            result = _evaluate(
                pipeline,
                k_value,
                seed=args.seed,
                workers=args.workers,
                batch_size=args.batch_size,
                sleep_seconds=args.sleep_seconds,
                max_retries=args.max_retries,
                backoff_seconds=args.backoff_seconds,
                backoff_multiplier=args.backoff_multiplier,
            )
        except DailyQuotaReached as error:
            LOGGER.warning(
                "Stopping RAGAS early at %s k=%d: %s",
                pipeline,
                k_value,
                error,
            )
            stopped_for_daily_quota = True
            break

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
                )
                in keys,
                axis=1,
            )
        ]

        merged = pd.concat(
            [existing, new_rows],
            ignore_index=True,
        )
    elif output_path.exists():
        merged = pd.read_csv(output_path)
    else:
        merged = pd.DataFrame(
            columns=[
                "pipeline",
                "k",
                "n",
                *RAGAS_COLUMNS,
            ]
        )

    if not merged.empty:
        merged = merged.sort_values(["pipeline", "k"]).reset_index(drop=True)

    merged.to_csv(output_path, index=False)

    LOGGER.info(
        "Wrote %s (%d rows)",
        output_path,
        len(merged),
    )

    if stopped_for_daily_quota:
        LOGGER.warning(
            (
                "Stopped cleanly after daily quota was reached. "
                "Rerun this command after the Gemini RPD reset to continue."
            )
        )


if __name__ == "__main__":
    main()
