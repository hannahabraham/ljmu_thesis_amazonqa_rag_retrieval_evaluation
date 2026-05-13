"""Unit tests for hallucination and refusal evaluation metrics."""

from __future__ import annotations

import math

import pandas as pd

from src.evaluation.hallucination import (
    hallucination_rate,
    refusal_rate_on_answerable,
)


def _frame(rows: list[dict]) -> pd.DataFrame:
    """Create a DataFrame from a list of row dictionaries."""
    return pd.DataFrame(rows)


def test_hallucination_uses_attempts_only() -> None:
    """Test hallucination rate excludes refusals and unanswerable rows."""
    dataframe = _frame(
        [
            {
                "is_answerable": True,
                "refused": False,
                "faithfulness": 0.9,
            },
            {
                "is_answerable": True,
                "refused": False,
                "faithfulness": 0.7,
            },
            {
                "is_answerable": True,
                "refused": True,
                "faithfulness": 0.1,
            },  # Excluded: refused
            {
                "is_answerable": False,
                "refused": True,
                "faithfulness": 1.0,
            },  # Excluded: unanswerable
        ]
    )

    rate = hallucination_rate(dataframe)

    # mean(1 - 0.9, 1 - 0.7) == 0.2
    assert abs(rate - 0.2) < 1e-9


def test_hallucination_empty_returns_nan() -> None:
    """Test hallucination rate returns NaN when no valid rows exist."""
    dataframe = _frame(
        [
            {
                "is_answerable": False,
                "refused": True,
                "faithfulness": 0.5,
            },
        ]
    )

    assert math.isnan(hallucination_rate(dataframe))


def test_hallucination_no_faithfulness_returns_nan() -> None:
    """Test hallucination rate returns NaN when faithfulness is missing."""
    dataframe = _frame(
        [
            {
                "is_answerable": True,
                "refused": False,
            },
        ]
    )

    assert math.isnan(hallucination_rate(dataframe))


def test_refusal_rate_on_answerable() -> None:
    """Test refusal rate is computed only on answerable rows."""
    dataframe = _frame(
        [
            {"is_answerable": True, "refused": True},
            {"is_answerable": True, "refused": False},
            {"is_answerable": True, "refused": False},
            {"is_answerable": False, "refused": True},  # Ignored
        ]
    )

    expected_rate = 1.0 / 3.0

    assert abs(refusal_rate_on_answerable(dataframe) - expected_rate) < 1e-9


def test_refusal_rate_no_answerable_rows() -> None:
    """Test refusal rate returns NaN when no answerable rows exist."""
    dataframe = _frame(
        [
            {
                "is_answerable": False,
                "refused": True,
            },
        ]
    )

    assert math.isnan(refusal_rate_on_answerable(dataframe))