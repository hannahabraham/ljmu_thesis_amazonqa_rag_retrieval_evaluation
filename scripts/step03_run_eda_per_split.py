"""Run per-split EDA and generate summary outputs and plots."""

from __future__ import annotations

import json
import logging

import pandas as pd

from config.settings import DATASET_FILES, PROCESSED_DIR
from src.eda import make_plots, summarise_split
from src.preprocessing import LIST_FIELDS
from src.utils.io import parse_list_field
from src.utils.logging_config import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Generate EDA summaries and plots for each dataset split."""
    summary_rows: list[dict] = []

    for split_name in DATASET_FILES:
        dataset_path = PROCESSED_DIR / f"std_{split_name}.csv"

        if not dataset_path.exists():
            LOGGER.warning(
                "Missing %s; run preprocessing step first",
                dataset_path,
            )
            continue

        dataframe = pd.read_csv(dataset_path)

        for field_name in LIST_FIELDS:
            if field_name in dataframe.columns:
                dataframe[field_name] = dataframe[field_name].apply(
                    parse_list_field
                )

        summary = summarise_split(dataframe, split_name)

        summary_rows.append(summary)

        make_plots(dataframe, split_name)

    summary_df = pd.DataFrame(summary_rows)

    if "qtype_counts" in summary_df.columns:
        summary_df["qtype_counts"] = summary_df["qtype_counts"].apply(
            json.dumps
        )

    output_path = PROCESSED_DIR / "eda_summary.csv"

    summary_df.to_csv(output_path, index=False)

    LOGGER.info("Wrote EDA summary to %s", output_path)


if __name__ == "__main__":
    main()
