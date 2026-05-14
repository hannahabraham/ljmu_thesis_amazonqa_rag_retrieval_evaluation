"""Validate refusal detection against manually labelled samples.

Workflow:
    1. Run a 50-sample smoke generation using any pipeline and k value.
    2. Open ``outputs/refusal_validation_set.csv``.
    3. Add a ``human_label`` column where 1 = refusal and 0 = not refusal.
    4. Re-run this script to compute precision, recall, and F1.
    5. If precision or recall is below 0.95, update refusal patterns and retry.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from config.settings import OUTPUT_DIR
from src.generation.refusal import is_refusal
from src.utils.logging_config import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)

MIN_ACCEPTABLE_SCORE = 0.95


def validate_refusal_detector(path: Path) -> None:
    """Validate the refusal detector on a labelled CSV file."""
    dataframe = pd.read_csv(path)

    if "human_label" not in dataframe.columns:
        raise SystemExit(
            "human_label column missing. Please label samples first."
        )

    dataframe["predicted"] = (
        dataframe["generated_answer"]
        .apply(is_refusal)
        .astype(int)
    )

    precision = precision_score(
        dataframe["human_label"],
        dataframe["predicted"],
    )
    recall = recall_score(
        dataframe["human_label"],
        dataframe["predicted"],
    )
    f1 = f1_score(
        dataframe["human_label"],
        dataframe["predicted"],
    )

    LOGGER.info("Precision: %.3f", precision)
    LOGGER.info("Recall:    %.3f", recall)
    LOGGER.info("F1:        %.3f", f1)

    if precision < MIN_ACCEPTABLE_SCORE or recall < MIN_ACCEPTABLE_SCORE:
        raise SystemExit(
            "Detector below threshold "
            f"(P={precision:.3f}, R={recall:.3f}). "
            "Edit src/generation/refusal.py patterns."
        )


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Validate refusal detection against human labels.",
    )

    parser.add_argument(
        "--path",
        type=Path,
        default=OUTPUT_DIR / "refusal_validation_set.csv",
        help="Path to the labelled refusal validation CSV.",
    )

    return parser


def main() -> None:
    """Parse CLI arguments and run validation."""
    parser = _build_parser()
    args = parser.parse_args()

    validate_refusal_detector(args.path)


if __name__ == "__main__":
    main()