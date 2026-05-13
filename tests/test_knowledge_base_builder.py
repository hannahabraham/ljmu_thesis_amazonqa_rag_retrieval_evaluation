"""Unit tests for knowledge base construction."""

from __future__ import annotations

import pandas as pd

from src.knowledge_base_builder import build_knowledge_base


def test_kb_keeps_all_reviews_no_cap(sample_record: dict) -> None:
    """Test all valid review snippets are retained without truncation."""
    dataframe = pd.DataFrame([sample_record])

    knowledge_base = build_knowledge_base(dataframe)

    # Three distinct review snippets, each above the minimum token threshold.
    assert len(knowledge_base) == 3
    assert knowledge_base["doc_id"].is_unique

    assert all(
        text.startswith("Product 1")
        or text.startswith("I dropped")
        or text.startswith("Battery")
        for text in knowledge_base["review_text"]
    )


def test_kb_dedupes_within_record(sample_record: dict) -> None:
    """Test duplicate review snippets are removed within a single record."""
    sample_record["review_snippets"] = [
        "Product 1 is fully waterproof to 10m.",
        "Product 1 is fully waterproof to 10m.",
        "Different review entirely about battery life.",
    ]

    dataframe = pd.DataFrame([sample_record])

    knowledge_base = build_knowledge_base(dataframe)

    assert len(knowledge_base) == 2


def test_kb_skips_short_reviews(sample_record: dict) -> None:
    """Test review snippets below the minimum token threshold are skipped."""
    sample_record["review_snippets"] = [
        "short",
        "ok!",
        "Long enough review with five words.",
    ]

    dataframe = pd.DataFrame([sample_record])

    knowledge_base = build_knowledge_base(dataframe)

    assert len(knowledge_base) == 1