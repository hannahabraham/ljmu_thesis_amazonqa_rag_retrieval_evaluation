"""Reproducibility drift between RANDOM_SEED and REPRO_SEED_2 at k=5.

Reads per-question JSONL from outputs/per_question/ (seed 1) and
outputs/per_question_seed2/ (seed 2), computes mean F1, faithfulness, and
answerability_accuracy per pipeline under each seed, and writes the absolute
drift to outputs/tables/reproducibility_drift.csv.

Rows where token_f1 drift exceeds F1_DRIFT_FLAG_THRESHOLD (2%) are flagged for
the thesis limitations chapter.
"""
from __future__ import annotations

import logging

from config.settings import (
    PER_QUESTION_DIR,
    PER_QUESTION_SEED2_DIR,
    PIPELINE_KEYS,
    RANDOM_SEED,
    REPRO_SEED_2,
    TABLES_DIR,
)
from src.evaluation.reproducibility import (
    F1_DRIFT_FLAG_THRESHOLD,
    reproducibility_drift,
)
from src.utils.io import load_per_question
from src.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    seed1 = load_per_question(
        PER_QUESTION_DIR, pipelines=list(PIPELINE_KEYS), ks=[5], seed=RANDOM_SEED,
    )
    seed2 = load_per_question(
        PER_QUESTION_SEED2_DIR, pipelines=list(PIPELINE_KEYS), ks=[5], seed=REPRO_SEED_2,
    )

    if seed1.empty or seed2.empty:
        logger.error(
            "Missing per-question JSONL for reproducibility check "
            "(seed1=%d rows, seed2=%d rows)",
            len(seed1), len(seed2),
        )
        return

    drift = reproducibility_drift(seed1, seed2)
    out = TABLES_DIR / "reproducibility_drift.csv"
    drift.to_csv(out, index=False)
    logger.info("Wrote %s (%d rows)", out, len(drift))

    flagged = drift[drift["flagged"]]
    if len(flagged):
        logger.warning(
            "Reproducibility drift > %.0f%% on token_f1 for: %s",
            F1_DRIFT_FLAG_THRESHOLD * 100.0,
            ", ".join(flagged["pipeline"].tolist()),
        )
    else:
        logger.info("No pipelines flagged (token_f1 drift below threshold).")


if __name__ == "__main__":
    main()
