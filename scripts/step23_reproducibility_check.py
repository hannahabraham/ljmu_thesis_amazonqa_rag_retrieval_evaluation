"""Compute reproducibility drift between two evaluation seeds at k=5."""

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
LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Load both seed runs, compute drift, and write the drift table."""
    seed_one = load_per_question(
        PER_QUESTION_DIR,
        pipelines=list(PIPELINE_KEYS),
        ks=[5],
        seed=RANDOM_SEED,
    )

    seed_two = load_per_question(
        PER_QUESTION_SEED2_DIR,
        pipelines=list(PIPELINE_KEYS),
        ks=[5],
        seed=REPRO_SEED_2,
    )

    if seed_one.empty or seed_two.empty:
        LOGGER.error(
            (
                "Missing per-question JSONL for reproducibility check "
                "(seed1=%d rows, seed2=%d rows)"
            ),
            len(seed_one),
            len(seed_two),
        )
        return

    drift = reproducibility_drift(
        seed_one,
        seed_two,
    )

    output_path = TABLES_DIR / "reproducibility_drift.csv"

    drift.to_csv(output_path, index=False)

    LOGGER.info(
        "Wrote %s (%d rows)",
        output_path,
        len(drift),
    )

    flagged = drift[drift["flagged"]]

    if not flagged.empty:
        LOGGER.warning(
            "Reproducibility drift > %.0f%% on token_f1 for: %s",
            F1_DRIFT_FLAG_THRESHOLD * 100.0,
            ", ".join(flagged["pipeline"].tolist()),
        )
    else:
        LOGGER.info(
            "No pipelines flagged; token_f1 drift is below threshold."
        )


if __name__ == "__main__":
    main()