"""Three chunking strategies operating on full reviews.

  - passage chunks  -> BM25, Dense, Hybrid    (200-token window, 20 overlap)
  - sentence chunks -> Sentence Window         (NLTK sent_tokenize + neighbours)
  - parent/child    -> Parent-Child            (full review = parent, ~100-tok children)
"""
from __future__ import annotations

import logging

import pandas as pd

from config.settings import (
    CHILD_CHUNK_TOKENS,
    PASSAGE_CHUNK_OVERLAP,
    PASSAGE_CHUNK_TOKENS,
)

logger = logging.getLogger(__name__)


def _ensure_nltk() -> None:
    import nltk

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:  # noqa: BLE001 -- older NLTK lacks punkt_tab
            pass


def _word_window_chunks(text: str, window: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= window:
        return [text]
    step = max(1, window - overlap)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        piece = words[start:start + window]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if start + window >= len(words):
            break
    return chunks


# ---------- Passage chunks ----------


def build_passage_chunks(kb_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    counter = 1
    for _, row in kb_df.iterrows():
        for chunk in _word_window_chunks(
            row["review_text"], PASSAGE_CHUNK_TOKENS, PASSAGE_CHUNK_OVERLAP
        ):
            rows.append({
                "chunk_id": f"PASS_{counter:06d}",
                "doc_id": row["doc_id"],
                "record_id": row["record_id"],
                "asin": row["asin"],
                "category": row["category"],
                "text": chunk,
                "n_words": len(chunk.split()),
            })
            counter += 1
    return pd.DataFrame(rows)


# ---------- Sentence chunks ----------


def build_sentence_chunks(kb_df: pd.DataFrame) -> pd.DataFrame:
    _ensure_nltk()
    from nltk.tokenize import sent_tokenize

    rows: list[dict] = []
    counter = 1
    for _, row in kb_df.iterrows():
        sentences = sent_tokenize(row["review_text"])
        local_ids: list[str] = []
        for sent in sentences:
            local_ids.append(f"SENT_{counter:06d}")
            counter += 1
        for i, (sent_id, sent) in enumerate(zip(local_ids, sentences)):
            rows.append({
                "chunk_id": sent_id,
                "doc_id": row["doc_id"],
                "record_id": row["record_id"],
                "asin": row["asin"],
                "category": row["category"],
                "text": sent,
                "sentence_index": i,
                "prev_sent_id": local_ids[i - 1] if i > 0 else None,
                "next_sent_id": local_ids[i + 1] if i + 1 < len(local_ids) else None,
                "n_words": len(sent.split()),
            })
    return pd.DataFrame(rows)


# ---------- Parent-child chunks ----------


def build_parent_child_chunks(kb_df: pd.DataFrame) -> pd.DataFrame:
    """Each KB review is a parent; children are ~CHILD_CHUNK_TOKENS slices."""
    rows: list[dict] = []
    counter = 1
    for _, row in kb_df.iterrows():
        children = _word_window_chunks(row["review_text"], CHILD_CHUNK_TOKENS, 0)
        for child_text in children:
            rows.append({
                "chunk_id": f"CHILD_{counter:06d}",
                "parent_id": row["doc_id"],
                "doc_id": row["doc_id"],
                "record_id": row["record_id"],
                "asin": row["asin"],
                "category": row["category"],
                "text": child_text,
                "parent_text": row["review_text"],
                "n_words": len(child_text.split()),
            })
            counter += 1
    return pd.DataFrame(rows)
