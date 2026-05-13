"""Unit tests for BM25 and hybrid retrieval utilities.

Qdrant-backed retrievers are covered in integration tests.
"""

from __future__ import annotations

import pandas as pd

from src.retrievers.bm25 import BM25Retriever
from src.retrievers.hybrid import rrf


def test_bm25_filters_by_asin() -> None:
    """Test BM25 retrieval filters results by ASIN."""
    chunks = pd.DataFrame(
        [
            {
                "chunk_id": "c1",
                "doc_id": "KB_00001",
                "asin": "A",
                "text": "waterproof phone",
            },
            {
                "chunk_id": "c2",
                "doc_id": "KB_00002",
                "asin": "B",
                "text": "waterproof phone",
            },
        ]
    )

    retriever = BM25Retriever(chunks, text_col="text")

    hits = list(
        retriever.retrieve(
            "is it waterproof",
            asin="A",
            k=5,
        )
    )

    assert all(hit["asin"] == "A" for hit in hits)
    assert hits
    assert hits[0]["chunk_id"] == "c1"


def test_bm25_returns_empty_when_no_asin_match() -> None:
    """Test BM25 retrieval returns no results for unmatched ASIN."""
    chunks = pd.DataFrame(
        [
            {
                "chunk_id": "c1",
                "doc_id": "KB_00001",
                "asin": "A",
                "text": "waterproof phone",
            },
        ]
    )

    retriever = BM25Retriever(chunks, text_col="text")

    results = list(
        retriever.retrieve(
            "anything",
            asin="ZZ",
            k=3,
        )
    )

    assert results == []


def test_rrf_fuses_two_rankings() -> None:
    """Test reciprocal rank fusion combines rankings correctly."""
    first_ranking = ["x", "y", "z"]
    second_ranking = ["y", "x", "w"]

    fused = rrf([first_ranking, second_ranking])

    fused_ids = [candidate for candidate, _ in fused]

    # x and y appear near the top in both rankings,
    # so they should outrank z and w.
    assert fused_ids[0] in {"x", "y"}
    assert "z" in fused_ids
    assert "w" in fused_ids