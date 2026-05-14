"""Create passage, sentence-window, and parent-child chunks from full reviews."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from config.settings import (
    CHILD_CHUNK_TOKENS,
    PASSAGE_CHUNK_OVERLAP,
    PASSAGE_CHUNK_TOKENS,
)

LOGGER = logging.getLogger(__name__)


def _ensure_nltk() -> None:
    """Ensure required NLTK tokenizers are available."""
    import nltk  # pylint: disable=import-outside-toplevel

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:  # pylint: disable=broad-exception-caught
            LOGGER.debug("punkt_tab is unavailable in this NLTK version")


def _word_window_chunks(
    text: str,
    window: int,
    overlap: int,
) -> list[str]:
    """Split text into word-window chunks."""
    words = text.split()

    if not words:
        return []

    if len(words) <= window:
        return [text]

    step = max(1, window - overlap)
    chunks: list[str] = []

    for start_index in range(0, len(words), step):
        piece = words[start_index:start_index + window]

        if not piece:
            break

        chunks.append(" ".join(piece))

        if start_index + window >= len(words):
            break

    return chunks


def _base_chunk_payload(
    row: pd.Series,
    chunk_id: str,
    text: str,
) -> dict[str, Any]:
    """Build common payload fields shared by all chunk types."""
    return {
        "chunk_id": chunk_id,
        "doc_id": row["doc_id"],
        "record_id": row["record_id"],
        "asin": row["asin"],
        "category": row["category"],
        "text": text,
        "n_words": len(text.split()),
    }


def build_passage_chunks(kb_df: pd.DataFrame) -> pd.DataFrame:
    """Build fixed-size passage chunks for BM25, dense, and hybrid retrieval."""
    rows: list[dict[str, Any]] = []
    counter = 1

    for _, row in kb_df.iterrows():
        chunks = _word_window_chunks(
            row["review_text"],
            PASSAGE_CHUNK_TOKENS,
            PASSAGE_CHUNK_OVERLAP,
        )

        for chunk_text in chunks:
            rows.append(
                _base_chunk_payload(
                    row,
                    f"PASS_{counter:06d}",
                    chunk_text,
                )
            )
            counter += 1

    return pd.DataFrame(rows)


def build_sentence_chunks(kb_df: pd.DataFrame) -> pd.DataFrame:
    """Build sentence chunks with previous and next sentence identifiers."""
    _ensure_nltk()

    from nltk.tokenize import sent_tokenize  # pylint: disable=import-outside-toplevel

    rows: list[dict[str, Any]] = []
    counter = 1

    for _, row in kb_df.iterrows():
        sentences = sent_tokenize(row["review_text"])
        local_ids: list[str] = []

        for _ in sentences:
            local_ids.append(f"SENT_{counter:06d}")
            counter += 1

        for index, (sentence_id, sentence_text) in enumerate(
            zip(local_ids, sentences)
        ):
            payload = _base_chunk_payload(
                row,
                sentence_id,
                sentence_text,
            )
            payload.update(
                {
                    "sentence_index": index,
                    "prev_sent_id": (
                        local_ids[index - 1]
                        if index > 0
                        else None
                    ),
                    "next_sent_id": (
                        local_ids[index + 1]
                        if index + 1 < len(local_ids)
                        else None
                    ),
                }
            )
            rows.append(payload)

    return pd.DataFrame(rows)


def build_parent_child_chunks(kb_df: pd.DataFrame) -> pd.DataFrame:
    """Build child chunks using each full review as the parent text."""
    rows: list[dict[str, Any]] = []
    counter = 1

    for _, row in kb_df.iterrows():
        child_chunks = _word_window_chunks(
            row["review_text"],
            CHILD_CHUNK_TOKENS,
            overlap=0,
        )

        for child_text in child_chunks:
            payload = _base_chunk_payload(
                row,
                f"CHILD_{counter:06d}",
                child_text,
            )
            payload.update(
                {
                    "parent_id": row["doc_id"],
                    "parent_text": row["review_text"],
                }
            )
            rows.append(payload)
            counter += 1

    return pd.DataFrame(rows)