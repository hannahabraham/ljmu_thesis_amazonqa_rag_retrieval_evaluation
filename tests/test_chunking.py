"""Unit tests for chunk-building utilities."""

from __future__ import annotations

import pandas as pd

from src.chunking import (
    build_parent_child_chunks,
    build_passage_chunks,
    build_sentence_chunks,
)


def _kb_with_long_review() -> pd.DataFrame:
    """Return a knowledge-base DataFrame with one long review."""
    long_text = " ".join(["word"] * 450)

    return pd.DataFrame(
        [
            {
                "doc_id": "KB_00001",
                "record_id": "REC_001",
                "asin": "A1",
                "category": "Electronics",
                "review_text": long_text,
            },
        ]
    )


def test_passage_chunks_window_and_overlap() -> None:
    """Test passage chunks respect the configured maximum window size."""
    knowledge_base = _kb_with_long_review()

    chunks = build_passage_chunks(knowledge_base)

    assert len(chunks) >= 2
    assert all(len(chunk.split()) <= 200 for chunk in chunks["text"])


def test_sentence_chunks_carry_neighbours() -> None:
    """Test sentence chunks include previous and next sentence identifiers."""
    knowledge_base = pd.DataFrame(
        [
            {
                "doc_id": "KB_00001",
                "record_id": "REC_001",
                "asin": "A1",
                "category": "Electronics",
                "review_text": (
                    "First sentence here. "
                    "Second sentence here. "
                    "Third sentence here."
                ),
            },
        ]
    )

    sentences = build_sentence_chunks(knowledge_base)

    # pandas coerces None to NaN in object/string columns when the DataFrame is
    # built from dict rows, so edge sentinels surface as NaN at the consumer.
    # SentenceWindowRetriever.get_window_ids handles both forms (None and NaN),
    # but downstream consumers should always use pd.isna() rather than identity.
    assert len(sentences) == 3
    assert pd.isna(sentences.iloc[0]["prev_sent_id"])
    assert not pd.isna(sentences.iloc[0]["next_sent_id"])
    assert pd.isna(sentences.iloc[-1]["next_sent_id"])


def test_parent_child_includes_full_parent_text() -> None:
    """Test parent-child chunks preserve full parent review text."""
    knowledge_base = _kb_with_long_review()

    parent_child_chunks = build_parent_child_chunks(knowledge_base)

    assert len(parent_child_chunks) >= 2
    assert (parent_child_chunks["parent_text"].str.split().str.len() == 450).all()
    assert parent_child_chunks["parent_id"].iloc[0] == "KB_00001"