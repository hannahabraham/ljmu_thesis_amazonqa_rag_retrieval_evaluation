"""Compute hallucination and refusal rates by pipeline and k value.

Hallucination rate is computed as mean ``1 - faithfulness`` over attempted
answers on answerable rows only. Refusals and gold-unanswerable rows are
excluded from hallucination-rate calculation.

If row-level faithfulness is unavailable, aggregate RAGAS faithfulness is used
as a fallback and the row is marked indicative.
"""

from __future__ import annotations

import logging

import pandas as pd

from config.settings import (
    K_VALUES,
    OUTPUT_DIR,
    PER_QUESTION_DIR,
    PIPELINE_KEYS,
    RANDOM_SEED,
    pipeline_output_dir,
)
from src.evaluation.hallucination import (
    hallucination_rate,
    refusal_rate_on_answerable,
)
from src.utils.io import load_per_question
from src.utils.logging_config import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)

RAGAS_COLUMNS = ["pipeline", "k", "faithfulness"]


def _aggregate_ragas() -> pd.DataFrame:
    """Load aggregate RAGAS metrics, or return an empty placeholder."""
    path = OUTPUT_DIR / "ragas_metrics.csv"

    if not path.exists():
        return pd.DataFrame(columns=RAGAS_COLUMNS)

    return pd.read_csv(path)


def _fallback_hallucination_rate(
    ragas_aggregate: pd.DataFrame,
    pipeline: str,
    k_value: int,
) -> float:
    """Compute fallback hallucination rate from aggregate faithfulness."""
    ragas_row = ragas_aggregate[
        (ragas_aggregate["pipeline"] == pipeline)
        & (ragas_aggregate["k"] == k_value)
    ]

    if ragas_row.empty:
        return float("nan")

    return 1.0 - float(ragas_row["faithfulness"].iloc[0])


def _compute_hallucination_row(
    per_question: pd.DataFrame,
    ragas_aggregate: pd.DataFrame,
    pipeline: str,
    k_value: int,
) -> dict:
    """Compute one hallucination/refusal metrics row."""
    has_row_level_faithfulness = (
        "faithfulness" in per_question.columns
        and per_question["faithfulness"].notna().any()
    )

    if has_row_level_faithfulness:
        hallucination = hallucination_rate(per_question)
        indicative = False
    else:
        hallucination = _fallback_hallucination_rate(
            ragas_aggregate,
            pipeline,
            k_value,
        )
        indicative = True

    refusal = refusal_rate_on_answerable(per_question)

    return {
        "pipeline": pipeline,
        "k": int(k_value),
        "n": int(len(per_question)),
        "hallucination_rate": (
            round(hallucination, 4)
            if pd.notna(hallucination)
            else ""
        ),
        "refusal_rate_on_answerable": (
            round(refusal, 4)
            if pd.notna(refusal)
            else ""
        ),
        "indicative": indicative,
    }


def _upsert_pipeline_metrics(
    pipeline: str,
    row: dict,
) -> None:
    """Write or update the per-pipeline hallucination metrics CSV."""
    output_path = pipeline_output_dir(pipeline) / "hallucination_metrics.csv"

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
    """Compute hallucination metrics and save aggregate outputs."""
    ragas_aggregate = _aggregate_ragas()
    rows: list[dict] = []

    for pipeline in PIPELINE_KEYS:
        for k_value in K_VALUES:
            per_question = load_per_question(
                PER_QUESTION_DIR,
                pipelines=[pipeline],
                ks=[k_value],
                seed=RANDOM_SEED,
            )

            if per_question.empty:
                LOGGER.warning(
                    "No per-question rows for %s k=%d",
                    pipeline,
                    k_value,
                )
                continue

            row = _compute_hallucination_row(
                per_question,
                ragas_aggregate,
                pipeline,
                k_value,
            )

            rows.append(row)

            _upsert_pipeline_metrics(
                pipeline,
                row,
            )

    output_path = OUTPUT_DIR / "hallucination_metrics.csv"

    pd.DataFrame(rows).to_csv(output_path, index=False)

    LOGGER.info(
        "Wrote %s (%d rows)",
        output_path,
        len(rows),
    )


if __name__ == "__main__":
    main()
