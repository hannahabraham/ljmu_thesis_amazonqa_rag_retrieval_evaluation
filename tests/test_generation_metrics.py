"""Unit tests for generation evaluation metrics."""

from __future__ import annotations

import pandas as pd

from src.evaluation.generation_metrics import (
    correct_answers_count,
    exact_match,
    is_correct,
    normalise_answer,
    token_f1,
    yesno_em,
)


def test_normalise_strips_articles_and_punct() -> None:
    """Test answer normalisation removes articles and punctuation."""
    assert normalise_answer("The Quick, Brown Fox.") == "quick brown fox"


def test_exact_match_after_normalisation() -> None:
    """Test exact match after answer normalisation."""
    assert exact_match("The fox", "a fox") == 1
    assert exact_match("fox", "dog") == 0


def test_token_f1_perfect_overlap() -> None:
    """Test token F1 returns 1.0 for perfect token overlap."""
    assert token_f1("battery lasts 8 hours", "the battery lasts 8 hours") == 1.0


def test_token_f1_partial() -> None:
    """Test token F1 returns a partial score for partial overlap."""
    score = token_f1("battery lasts 8", "battery lasts ten hours")

    assert 0.0 < score < 1.0


def test_token_f1_empty_match() -> None:
    """Test token F1 behaviour for empty predictions and references."""
    assert token_f1("", "") == 1.0
    assert token_f1("anything", "") == 0.0


def test_yesno_em() -> None:
    """Test yes/no exact-match behaviour."""
    assert yesno_em("Yes, it works.", "yes") == 1
    assert yesno_em("No, it doesn't.", "yes") == 0


def test_is_correct_answerable_above_threshold() -> None:
    """Test answerable non-refusal is correct above the F1 threshold."""
    row = {"is_answerable": True, "refused": False, "token_f1": 0.6}

    assert is_correct(row, f1_threshold=0.5) is True


def test_is_correct_answerable_below_threshold() -> None:
    """Test answerable non-refusal is incorrect below the F1 threshold."""
    row = {"is_answerable": True, "refused": False, "token_f1": 0.3}

    assert is_correct(row, f1_threshold=0.5) is False


def test_is_correct_answerable_refused() -> None:
    """Test answerable refusal is considered incorrect."""
    row = {"is_answerable": True, "refused": True, "token_f1": 0.9}

    assert is_correct(row, f1_threshold=0.5) is False


def test_is_correct_unanswerable_refused() -> None:
    """Test unanswerable refusal is considered correct."""
    row = {"is_answerable": False, "refused": True, "token_f1": 0.0}

    assert is_correct(row, f1_threshold=0.5) is True


def test_is_correct_unanswerable_answered() -> None:
    """Test unanswerable non-refusal is considered incorrect."""
    row = {"is_answerable": False, "refused": False, "token_f1": 1.0}

    assert is_correct(row, f1_threshold=0.5) is False


def test_correct_threshold_sensitivity() -> None:
    """Test correct answer count changes with the F1 threshold."""
    dataframe = pd.DataFrame(
        [
            {"is_answerable": True, "refused": False, "token_f1": 0.35},
            {"is_answerable": True, "refused": False, "token_f1": 0.55},
            {"is_answerable": True, "refused": False, "token_f1": 0.75},
            {"is_answerable": False, "refused": True, "token_f1": 0.0},
        ]
    )

    assert correct_answers_count(dataframe, threshold=0.3) == 4
    assert correct_answers_count(dataframe, threshold=0.5) == 3
    assert correct_answers_count(dataframe, threshold=0.7) == 2