"""Merge standardised dataset splits and remove invalid or duplicate rows."""

from __future__ import annotations

import logging

import pandas as pd

from config.settings import DATASET_FILES, PROCESSED_DIR
from src.utils.logging_config import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Merge standardised dataset splits into a single cleaned CSV."""
    dataframes: list[pd.DataFrame] = []

    for split_name in DATASET_FILES:
        dataset_path = PROCESSED_DIR / f"std_{split_name}.csv"

        if dataset_path.exists():
            dataframes.append(pd.read_csv(dataset_path))
        else:
            LOGGER.warning("Missing standardised split: %s", dataset_path)

    if not dataframes:
        LOGGER.warning("No standardised dataset files found")
        return

    combined_df = pd.concat(dataframes, ignore_index=True)

    before_count = len(combined_df)

    combined_df = combined_df.dropna(
        subset=["qid", "asin", "questionText"]
    )

    combined_df = combined_df.drop_duplicates(
        subset=["qid", "source_file"]
    )

    LOGGER.info(
        "After cleaning: %d -> %d rows",
        before_count,
        len(combined_df),
    )

    output_path = PROCESSED_DIR / "combined_amazonqa.csv"

    combined_df.to_csv(output_path, index=False)

    LOGGER.info("Wrote merged dataset to %s", output_path)


if __name__ == "__main__":
    main()