"""Unit tests for refusal detection utilities."""

from __future__ import annotations

import pytest

from src.generation.refusal import is_refusal


@pytest.mark.parametrize(
    "answer",
    [
        (
            "The available reviews do not provide enough information "
            "to answer this question."
        ),
        "I cannot determine from the reviews whether it is waterproof.",
        "The reviews don't say anything about battery life.",
        "There is no information in the context about this.",
        "Insufficient evidence in the reviews to answer.",
        "Unable to verify from the reviews.",
        "It is unclear from the reviews.",
        "Not enough information.",
        "",
        "   ",
    ],
)
def test_refusal_phrasings_detected(answer: str) -> None:
    """Test common refusal phrasings are correctly detected."""
    assert is_refusal(answer) is True


@pytest.mark.parametrize(
    "answer",
    [
        "Yes, it is waterproof to 10 meters.",
        (
            "The blender works well with frozen fruit "
            "according to several reviews."
        ),
        "No, the device does not support USB-C.",
        "Customers report battery life of around 8 hours.",
    ],
)
def test_factual_answers_not_refusal(answer: str) -> None:
    """Test factual answers are not incorrectly flagged as refusals."""
    assert is_refusal(answer) is False