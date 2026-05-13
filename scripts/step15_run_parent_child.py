"""Run the parent-child retrieval pipeline for one or more k values.

Writes per-question outputs using run IDs such as ``pc_k5_seed42``.

Examples:
    python -m scripts.step15_run_parent_child --ks 5
    python -m scripts.step15_run_parent_child --ks 1 3 5 10 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config.settings import RANDOM_SEED
from src.pipelines.runner import run_pipeline_cells
from src.utils.logging_config import configure_logging


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run the parent-child retrieval pipeline.",
    )

    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
        help="Retrieval depth values to evaluate.",
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional smoke-test sample size from the golden set.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Seed for run_id and supported generation components.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override per-question JSONL output directory.",
    )

    return parser


def main() -> None:
    """Parse CLI arguments and run the parent-child retrieval pipeline."""
    configure_logging()

    parser = _build_parser()
    args = parser.parse_args()

    run_pipeline_cells(
        "pc",
        args.ks,
        sample=args.sample,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()