"""Run a fail-fast consistency check for the verified golden dataset."""

from __future__ import annotations

import logging

import pandas as pd

from config.settings import PROCESSED_DIR
from src.golden_dataset_builder import validate_golden_consistency
from src.utils.logging_config import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Validate the verified golden dataset against the knowledge base."""
    golden_path = PROCESSED_DIR / "golden_dataset_200_verified.csv"
    knowledge_base_path = PROCESSED_DIR / "knowledge_base_full_reviews.csv"

    golden_dataset = pd.read_csv(golden_path)
    knowledge_base = pd.read_csv(knowledge_base_path)

    validate_golden_consistency(
        golden_dataset,
        knowledge_base,
    )

    LOGGER.info("Golden dataset is consistent with the knowledge base")


if __name__ == "__main__":
    main()