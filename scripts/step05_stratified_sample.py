"""Create a two-stage stratified 200-record AmazonQA sample.

The sample contains 120 train, 40 validation, and 40 test records.

Stage 1 floors each named category at ``MIN_PER_NAMED_CATEGORY`` records,
stratified within each category by ``questionType`` and ``is_answerable``.
Stage 2 fills the remainder from the wider dataset, also stratified.

Per-category counts are logged so Table 3 floors can be eye-verified.
"""

from __future__ import annotations

import logging

import pandas as pd

from config.settings import (
    MIN_PER_NAMED_CATEGORY,
    NAMED_CATEGORIES,
    PROCESSED_DIR,
    RANDOM_SEED,
)
from src.sampling import two_stage_stratified_sample
from src.utils.logging_config import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Create and persist the final stratified 200-record sample."""
    combined_path = PROCESSED_DIR / "combined_amazonqa.csv"
    output_path = PROCESSED_DIR / "final_200_records.csv"

    combined_df = pd.read_csv(combined_path, low_memory=False)

    sample = two_stage_stratified_sample(
        combined_df,
        seed=RANDOM_SEED,
    )

    sample.to_csv(output_path, index=False)

    LOGGER.info("Wrote %d rows to %s", len(sample), output_path)
    LOGGER.info("Total rows: %d", len(sample))

    counts = sample["category"].value_counts()

    for category in NAMED_CATEGORIES:
        count = int(counts.get(category, 0))
        status = (
            "OK"
            if count >= MIN_PER_NAMED_CATEGORY
            else "BELOW FLOOR"
        )

        LOGGER.info(
            "  %-30s n=%3d  [%s]",
            category,
            count,
            status,
        )

    other_categories = sample[
        ~sample["category"].isin(NAMED_CATEGORIES)
    ]

    if not other_categories.empty:
        LOGGER.info(
            "  Other categories total: n=%d",
            len(other_categories),
        )


if __name__ == "__main__":
    main()