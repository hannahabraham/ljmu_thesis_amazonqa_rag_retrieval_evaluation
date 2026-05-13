"""Build the BM25 index and Qdrant vector collections."""

from __future__ import annotations

import logging

import pandas as pd

from config.settings import PROCESSED_DIR
from src.indexing import build_all_qdrant_collections, build_bm25_index
from src.utils.logging_config import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Load chunk tables and build all retrieval indexes."""
    passage_chunks_path = PROCESSED_DIR / "passage_chunks.csv"
    sentence_chunks_path = PROCESSED_DIR / "sentence_chunks.csv"
    parent_child_chunks_path = PROCESSED_DIR / "parent_child_chunks.csv"

    passage_chunks = pd.read_csv(passage_chunks_path)
    sentence_chunks = pd.read_csv(sentence_chunks_path)
    parent_child_chunks = pd.read_csv(parent_child_chunks_path)

    build_bm25_index(passage_chunks)

    build_all_qdrant_collections(
        passage_chunks,
        sentence_chunks,
        parent_child_chunks,
    )

    LOGGER.info("Built BM25 index and Qdrant collections")


if __name__ == "__main__":
    main()