"""Build a full-review knowledge base from the final sampled records."""

from __future__ import annotations

import logging

import pandas as pd

from config.settings import PROCESSED_DIR
from src.knowledge_base_builder import build_knowledge_base
from src.utils.logging_config import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Build and save the retrieval knowledge base."""
    input_path = PROCESSED_DIR / "final_200_records.csv"
    output_path = PROCESSED_DIR / "knowledge_base_full_reviews.csv"

    final_records = pd.read_csv(input_path)

    knowledge_base = build_knowledge_base(final_records)

    knowledge_base.to_csv(output_path, index=False)

    LOGGER.info(
        "Wrote %d knowledge-base rows to %s",
        len(knowledge_base),
        output_path,
    )


if __name__ == "__main__":
    main()