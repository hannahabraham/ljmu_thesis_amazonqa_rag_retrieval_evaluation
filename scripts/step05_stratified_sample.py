"""Create the quota-stratified 200-record AmazonQA sample.

The sample is drawn against an explicit ``(questionType, is_answerable,
split)`` quota table (``SAMPLE_QUOTAS`` in ``config/settings.py``) so that
yes/no and unanswerable cells have enough rows to power their sub-analyses.
Proportional sampling under-represents both (yes/no is ~15% of the
population, unanswerable is ~38%).

Targets:
    yes/no       = 55  (35 answerable + 20 unanswerable)
    descriptive  = 145 (90 answerable + 55 unanswerable)
    train/val/test = 120/40/40
"""

from __future__ import annotations

import logging

import pandas as pd

from config.settings import NAMED_CATEGORIES, PROCESSED_DIR, RANDOM_SEED
from src.sampling import quota_stratified_sample
from src.utils.logging_config import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Create and persist the final quota-stratified sample."""
    combined_path = PROCESSED_DIR / "combined_amazonqa.csv"
    output_path = PROCESSED_DIR / "final_200_records.csv"

    combined_df = pd.read_csv(combined_path, low_memory=False)

    sample = quota_stratified_sample(combined_df, seed=RANDOM_SEED)

    sample.to_csv(output_path, index=False)

    LOGGER.info("Wrote %d rows to %s", len(sample), output_path)

    LOGGER.info("Balance check:")
    LOGGER.info(
        "  questionType:\n%s",
        sample["questionType"].value_counts().to_string(),
    )
    LOGGER.info(
        "  is_answerable:\n%s",
        sample["is_answerable"].value_counts().to_string(),
    )
    LOGGER.info(
        "  source_file:\n%s",
        sample["source_file"].value_counts().to_string(),
    )

    counts = sample["category"].value_counts()
    LOGGER.info("Named-category coverage (informational; quotas drive selection):")
    for category in NAMED_CATEGORIES:
        LOGGER.info("  %-30s n=%3d", category, int(counts.get(category, 0)))

    other_categories = sample[~sample["category"].isin(NAMED_CATEGORIES)]
    if not other_categories.empty:
        LOGGER.info(
            "  Other categories total: n=%d",
            len(other_categories),
        )


if __name__ == "__main__":
    main()
