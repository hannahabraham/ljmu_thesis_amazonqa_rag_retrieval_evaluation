"""Bundle Results Sheet CSV files into one Excel workbook."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config.settings import OUTPUT_DIR, TABLES_DIR
from src.utils.logging_config import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)

SHEETS: dict[str, Path] = {
    "Table1_Overall": TABLES_DIR / "table1_overall.csv",
    "Table2_Depth": TABLES_DIR / "table2_depth.csv",
    "Table3_Category": TABLES_DIR / "table3_category.csv",
    "Table4_Length": TABLES_DIR / "table4_length.csv",
    "Table6_Answerability": TABLES_DIR / "table6_answerability.csv",
    "Table7_Final_Ranking": TABLES_DIR / "table7_final_ranking.csv",
    "Pairwise_Wilcoxon": TABLES_DIR / "pairwise_wilcoxon.csv",
    "RAGAS_Raw": OUTPUT_DIR / "ragas_metrics.csv",
    "Correct_Threshold_Sweep": OUTPUT_DIR / "correct_threshold_sensitivity.csv",
    "Composite_Weight_Sweep": OUTPUT_DIR / "composite_weight_sensitivity.csv",
}


def main() -> None:
    """Write available result CSVs into a multi-sheet Excel workbook."""
    output_path = OUTPUT_DIR / "thesis_results.xlsx"
    written_sheets = 0

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, csv_path in SHEETS.items():
            if not csv_path.exists():
                LOGGER.warning(
                    "Skipping %s because %s is missing",
                    sheet_name,
                    csv_path,
                )
                continue

            dataframe = pd.read_csv(csv_path)

            dataframe.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )

            written_sheets += 1

    if written_sheets == 0:
        LOGGER.error(
            "No source CSV files found. Run scripts 22 and 24 first."
        )
        return

    LOGGER.info(
        "Wrote %s with %d sheets",
        output_path,
        written_sheets,
    )


if __name__ == "__main__":
    main()
