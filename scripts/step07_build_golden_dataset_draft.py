"""Build the draft golden dataset using grounding and Jeffreys scoring."""

from __future__ import annotations

import logging

import pandas as pd

from config.settings import PROCESSED_DIR
from src.golden_dataset_builder import build_golden_draft
from src.utils.logging_config import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Build and persist the draft golden dataset."""
    final_records_path = (
        PROCESSED_DIR / "final_200_records.csv"
    )

    knowledge_base_path = (
        PROCESSED_DIR / "knowledge_base_full_reviews.csv"
    )

    output_path = (
        PROCESSED_DIR / "golden_dataset_200_draft.csv"
    )

    final_records = pd.read_csv(final_records_path)

    knowledge_base = pd.read_csv(knowledge_base_path)

    draft_dataset = build_golden_draft(
        final_records,
        knowledge_base,
    )

    draft_dataset.to_csv(output_path, index=False)

    flagged_count = int(draft_dataset["needs_judge"].sum())

    LOGGER.info(
        "Draft golden dataset: %d rows, %d flagged for judge review",
        len(draft_dataset),
        flagged_count,
    )


if __name__ == "__main__":
    main()
