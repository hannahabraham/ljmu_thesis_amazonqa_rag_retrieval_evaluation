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


@pytest.mark.parametrize(
    "answer",
    [
        # Yes/No leads protect against later disclaimer phrases.
        "Yes, the reviews don't specify the exact length but most say 6 feet.",
        "No, although the reviews do not mention the precise weight.",
        # Disclaimer mid-sentence after factual leading clause.
        (
            "Customers report battery life of around 8 hours, "
            "although the reviews don't mention waterproofing."
        ),
        (
            "The cable is roughly 4 feet long; the reviews don't say the "
            "exact tolerance."
        ),
        # Factual statement using "no" as a quantifier elsewhere.
        "The transmitter works with any 3.5 mm headphone jack.",
        # "Don't" used factually about reviewers, not as a refusal.
        "Some reviewers don't love it but most rate it 4 stars or higher.",
    ],
)
def test_adversarial_factuals_not_refusal(answer: str) -> None:
    """Factual answers that mention a gap mid-sentence must not refuse."""
    assert is_refusal(answer) is False


@pytest.mark.parametrize(
    "answer",
    [
        # Refusals embedded as the full first clause.
        (
            "The reviews don't mention battery life, "
            "so I cannot answer that."
        ),
        "Unclear from the reviews.",
        "No information in the reviews about waterproofing.",
        "Not specified in the reviews.",
    ],
)
def test_real_refusals_still_detected(answer: str) -> None:
    """Genuine refusals must still be flagged after the leading-clause fix."""
    assert is_refusal(answer) is True