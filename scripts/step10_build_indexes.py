"""Build the Qdrant collections and persist the BM25 index.

Writes three Qdrant collections (passages, sentences, child_chunks) and a
``bm25.pkl`` pickle of :class:`src.retrievers.bm25.BM25Retriever` so the
pipeline runner can skip the in-memory rebuild on subsequent runs.
"""

from __future__ import annotations

import logging

import pandas as pd

from config.settings import BM25_PICKLE_PATH, PROCESSED_DIR
from src.indexing import build_all_qdrant_collections
from src.retrievers.bm25 import BM25Retriever
from src.utils.logging_config import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Load chunk tables, build Qdrant collections, and pickle the BM25 index."""
    passage_chunks_path = PROCESSED_DIR / "passage_chunks.csv"
    sentence_chunks_path = PROCESSED_DIR / "sentence_chunks.csv"
    parent_child_chunks_path = PROCESSED_DIR / "parent_child_chunks.csv"

    passage_chunks = pd.read_csv(passage_chunks_path)
    sentence_chunks = pd.read_csv(sentence_chunks_path)
    parent_child_chunks = pd.read_csv(parent_child_chunks_path)

    build_all_qdrant_collections(
        passage_chunks,
        sentence_chunks,
        parent_child_chunks,
    )
    LOGGER.info("Built Qdrant collections")

    bm25 = BM25Retriever(passage_chunks, text_col="text")
    bm25.save(BM25_PICKLE_PATH)
    LOGGER.info("Pickled BM25 index to %s", BM25_PICKLE_PATH)


if __name__ == "__main__":
    main()
