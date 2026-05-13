"""Bundle all metric CSVs into one .xlsx with the Sheet Tables."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config.settings import OUTPUT_DIR
from src.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

SHEETS = {
    "Table1_Overall": "generation_metrics.csv",  # filtered to k=5 in writer
    "Table2_Depth": "retrieval_metrics.csv",
    "Table3_Category": "category_metrics.csv",
    "Table4_QBucket": "qbucket_metrics.csv",
    "Table6_Answerability": "answerability_metrics.csv",
    "Table7_Final_Ranking": "final_ranking.csv",
    "RAGAS": "ragas_metrics.csv",
}


def main() -> None:
    out_path = OUTPUT_DIR / "thesis_results.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for sheet, filename in SHEETS.items():
            csv_path = Path(OUTPUT_DIR / filename)
            if not csv_path.exists():
                logger.warning("Skipping %s -- %s missing", sheet, csv_path)
                continue
            df = pd.read_csv(csv_path)
            if sheet == "Table1_Overall" and "k" in df.columns:
                df = df[df["k"] == 5]
            df.to_excel(writer, sheet_name=sheet, index=False)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
