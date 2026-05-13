"""Create passage, sentence, and parent-child chunk tables."""

from __future__ import annotations

import logging

import pandas as pd

from config.settings import PROCESSED_DIR
from src.chunking import (
    build_parent_child_chunks,
    build_passage_chunks,
    build_sentence_chunks,
)
from src.golden_dataset_builder import validate_golden_consistency
from src.utils.logging_config import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Build and persist all chunking variants."""
    knowledge_base_path = (
        PROCESSED_DIR / "knowledge_base_full_reviews.csv"
    )

    golden_dataset_path = (
        PROCESSED_DIR / "golden_dataset_200_verified.csv"
    )

    knowledge_base = pd.read_csv(knowledge_base_path)

    if golden_dataset_path.exists():
        golden_dataset = pd.read_csv(golden_dataset_path)

        validate_golden_consistency(
            golden_dataset,
            knowledge_base,
        )
    else:
        LOGGER.warning(
            "Verified golden dataset not found; "
            "skipping consistency precheck"
        )

    passage_chunks = build_passage_chunks(knowledge_base)

    sentence_chunks = build_sentence_chunks(knowledge_base)

    parent_child_chunks = build_parent_child_chunks(
        knowledge_base
    )

    passage_chunks.to_csv(
        PROCESSED_DIR / "passage_chunks.csv",
        index=False,
    )

    sentence_chunks.to_csv(
        PROCESSED_DIR / "sentence_chunks.csv",
        index=False,
    )

    parent_child_chunks.to_csv(
        PROCESSED_DIR / "parent_child_chunks.csv",
        index=False,
    )

    LOGGER.info(
        (
            "Created chunk tables: "
            "%d passages, %d sentences, %d parent-child"
        ),
        len(passage_chunks),
        len(sentence_chunks),
        len(parent_child_chunks),
    )


if __name__ == "__main__":
    main()