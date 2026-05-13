"""Shared pytest fixtures."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest


@pytest.fixture
def sample_record() -> dict[str, Any]:
    """Return a sample test record."""
    return {
        "record_id": "REC_001",
        "qid": "Q_FAKE_1",
        "asin": "B0FAKE1",
        "category": "Electronics",
        "source_file": "test",
        "questionType": "yesno",
        "is_answerable": 1,
        "questionText": "Is product 1 waterproof?",
        "review_snippets": [
            "Product 1 is fully waterproof to 10m.",
            "I dropped product 1 in the pool, it survived.",
            "Battery on product 1 lasts 8 hours.",
        ],
        "top_sentences_IR": [],
        "top_review_helpful": [],
        "top_review_wilson": [],
        "answers": [
            {
                "answerText": "Yes, waterproof to 10m.",
                "helpful": 12,
                "unhelpful": 1,
            },
            {
                "answerText": "I'm not sure.",
                "helpful": 0,
                "unhelpful": 2,
            },
        ],
    }


@pytest.fixture
def sample_record_df(sample_record: dict[str, Any]) -> pd.DataFrame:
    """Return a DataFrame containing a sample record."""
    return pd.DataFrame([sample_record])


@pytest.fixture
def fake_kb() -> pd.DataFrame:
    """Return a fake knowledge base DataFrame."""
    return pd.DataFrame(
        [
            {
                "doc_id": "KB_00001",
                "record_id": "REC_001",
                "qid": "Q_FAKE_1",
                "asin": "B0FAKE1",
                "category": "Electronics",
                "source_file": "test",
                "review_text": "Product 1 is fully waterproof to 10m.",
                "n_words": 7,
                "source_field": "review_snippets",
            },
            {
                "doc_id": "KB_00002",
                "record_id": "REC_001",
                "qid": "Q_FAKE_1",
                "asin": "B0FAKE1",
                "category": "Electronics",
                "source_file": "test",
                "review_text": "Battery lasts 8 hours.",
                "n_words": 4,
                "source_field": "review_snippets",
            },
        ]
    )


@pytest.fixture
def fake_groq() -> MagicMock:
    """Return a mocked Groq client."""
    mock_client = MagicMock()
    mock_client.invoke.return_value = (
        "Yes, this product is waterproof to 10 metres."
    )
    return mock_client


@pytest.fixture
def make_fake_llm_client():
    """Return a factory for creating mocked LLM clients."""

    def _factory(response: Any) -> MagicMock:
        """Create a MagicMock with a predefined invoke response."""
        mock_client = MagicMock()
        mock_client.invoke.return_value = response
        return mock_client

    return _factory