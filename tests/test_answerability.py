"""Unit tests for answerability evaluation."""

from __future__ import annotations

import pandas as pd

from src.evaluation.answerability import (
    classify_answerability,
    compute_answerability_table,
)


def test_classify_each_quadrant() -> None:
    """Test all answerability classification scenarios."""
    assert classify_answerability(1, False) == "correctly_answered"
    assert classify_answerability(1, True) == "wrongly_refused"
    assert classify_answerability(0, True) == "correctly_refused"
    assert classify_answerability(0, False) == "wrongly_answered"


def test_table_aggregates() -> None:
    """Test aggregate metrics in the answerability table."""
    df = pd.DataFrame(
        [
            {"is_answerable": 1, "refused": False},
            {"is_answerable": 1, "refused": True},
            {"is_answerable": 0, "refused": True},
            {"is_answerable": 0, "refused": False},
        ]
    )

    table = compute_answerability_table(df)

    assert int(table["n"].iloc[0]) == 4
    assert int(table["correctly_answered"].iloc[0]) == 1
    assert int(table["correctly_refused"].iloc[0]) == 1
    assert table["answerability_acc"].iloc[0] == 0.5