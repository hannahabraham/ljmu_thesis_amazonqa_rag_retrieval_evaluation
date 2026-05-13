"""Load JSONL dataset splits and persist standardised CSV files for EDA."""

from __future__ import annotations

import logging

from config.settings import DATASET_FILES, PROCESSED_DIR
from src.data_loader import load_jsonl
from src.preprocessing import standardize_split
from src.utils.logging_config import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Load, standardise, and persist dataset splits."""
    for split_name, dataset_path in DATASET_FILES.items():
        if not dataset_path.exists():
            LOGGER.warning("Missing dataset file: %s", dataset_path)
            continue

        dataframe = load_jsonl(dataset_path)

        standardised = standardize_split(
            dataframe,
            source_file=split_name,
        )

        output_path = PROCESSED_DIR / f"std_{split_name}.csv"

        standardised.to_csv(output_path, index=False)

        LOGGER.info(
            "Wrote %d rows to %s",
            len(standardised),
            output_path,
        )


if __name__ == "__main__":
    main()