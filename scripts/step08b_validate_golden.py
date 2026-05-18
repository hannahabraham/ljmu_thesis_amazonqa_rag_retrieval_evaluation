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
    """Validate the verified golden dataset against the knowledge base.

    Hard integrity failures (missing/dangling evidence, text drift) raise.
    Soft label mismatches are logged and written back to a
    ``validation_status`` column on the verified CSV, where step08c
    picks them up for manual review.
    """
    golden_path = PROCESSED_DIR / "golden_dataset_200_verified.csv"
    knowledge_base_path = PROCESSED_DIR / "knowledge_base_full_reviews.csv"

    golden_dataset = pd.read_csv(golden_path)
    knowledge_base = pd.read_csv(knowledge_base_path)

    annotated = validate_golden_consistency(
        golden_dataset,
        knowledge_base,
    )

    annotated.to_csv(golden_path, index=False)
    LOGGER.info(
        "Golden dataset passes hard integrity checks; "
        "validation_status column written to %s",
        golden_path,
    )


if __name__ == "__main__":
    main()
