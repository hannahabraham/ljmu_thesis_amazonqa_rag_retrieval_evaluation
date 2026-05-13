"""Hallucination + refusal-rate pair (v5).

Reported as a pair in Table 6 to expose the trade-off honestly: a pipeline that
refuses every answerable question has zero hallucination but is useless.

``hallucination_rate``: mean (1 - faithfulness) over **answerable rows where
the model attempted an answer**. Refusals can't hallucinate; gold-unanswerable
rows aren't subject to a faithfulness check the same way.

``refusal_rate_on_answerable``: refusals divided by answerable rows. Pair this
with ``hallucination_rate`` whenever you report either.
"""
from __future__ import annotations

import math

import pandas as pd

from src.evaluation.generation_metrics import _coerce_bool


def _bool_series(series: pd.Series) -> pd.Series:
    return series.apply(_coerce_bool)


def hallucination_rate(
    per_q: pd.DataFrame,
    faithfulness_per_row: pd.Series | None = None,
) -> float:
    """Mean (1 - faithfulness) over attempted answerable rows.

    `faithfulness_per_row` must align row-wise with `per_q`. When None and the
    DataFrame already has a ``faithfulness`` column, that column is used.
    Returns NaN when there are no qualifying rows.
    """
    if per_q.empty:
        return float("nan")
    answerable = _bool_series(per_q["is_answerable"])
    refused = _bool_series(per_q["refused"])
    mask = answerable & (~refused)
    if not mask.any():
        return float("nan")

    if faithfulness_per_row is None:
        if "faithfulness" not in per_q.columns:
            return float("nan")
        faithfulness_per_row = per_q["faithfulness"]

    aligned = pd.to_numeric(faithfulness_per_row, errors="coerce").reset_index(drop=True)
    subset = aligned.loc[mask.reset_index(drop=True)].dropna()
    if subset.empty:
        return float("nan")
    return float((1.0 - subset).mean())


def refusal_rate_on_answerable(per_q: pd.DataFrame) -> float:
    """Fraction of answerable rows where the model refused."""
    if per_q.empty:
        return float("nan")
    answerable = _bool_series(per_q["is_answerable"])
    if not answerable.any():
        return float("nan")
    refused = _bool_series(per_q["refused"])
    return float((refused & answerable).sum()) / float(answerable.sum())


def hallucination_refusal_pair(
    per_q: pd.DataFrame, faithfulness_per_row: pd.Series | None = None,
) -> dict[str, float]:
    """Convenience: return both metrics as a dict."""
    return {
        "hallucination_rate": hallucination_rate(per_q, faithfulness_per_row),
        "refusal_rate_on_answerable": refusal_rate_on_answerable(per_q),
    }


def is_nan(value: float) -> bool:
    return isinstance(value, float) and math.isnan(value)
