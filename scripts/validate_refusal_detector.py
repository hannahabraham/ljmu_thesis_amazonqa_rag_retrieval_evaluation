"""Validate the refusal detector against hand labels.

Workflow:
  1. Run a 50-sample smoke generation (any pipeline, any k).
  2. Open `outputs/refusal_validation_set.csv`, add `human_label` column (0 or 1).
  3. Re-run this script. It loads labels and prints precision/recall.
  4. If precision or recall < 0.95, edit src/generation/refusal.py patterns and repeat.
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from config.settings import OUTPUT_DIR
from src.generation.refusal import is_refusal
from src.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def main(path: str) -> None:
    df = pd.read_csv(path)
    if "human_label" not in df.columns:
        raise SystemExit("human_label column missing -- please label samples first")
    df["predicted"] = df["generated_answer"].apply(is_refusal).astype(int)
    p = precision_score(df["human_label"], df["predicted"])
    r = recall_score(df["human_label"], df["predicted"])
    f = f1_score(df["human_label"], df["predicted"])
    logger.info("Precision: %.3f", p)
    logger.info("Recall:    %.3f", r)
    logger.info("F1:        %.3f", f)
    if p < 0.95 or r < 0.95:
        raise SystemExit(f"Detector below threshold (P={p:.3f}, R={r:.3f}). Edit patterns.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=str(OUTPUT_DIR / "refusal_validation_set.csv"))
    main(parser.parse_args().path)
