"""Hallucination rate + refusal rate on answerable per (pipeline, k).

v5: reports the pair `(hallucination_rate, refusal_rate_on_answerable)` for
Table 6. Hallucination rate is mean (1 - faithfulness) over **attempted answers
on answerable rows only** (refusals don't count, gold-unanswerable doesn't count).

Requires per-row faithfulness in the per-question JSONL — populated by
scripts/18_eval_ragas.py. Falls back to the aggregate from
outputs/ragas_metrics.csv if per-row scores are absent (with a [indicative] flag).
"""
from __future__ import annotations

import logging

import pandas as pd

from config.settings import (
    K_VALUES,
    OUTPUT_DIR,
    PER_QUESTION_DIR,
    PIPELINE_KEYS,
    RANDOM_SEED,
    pipeline_output_dir,
)
from src.evaluation.hallucination import (
    hallucination_rate,
    refusal_rate_on_answerable,
)
from src.utils.io import load_per_question
from src.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def _aggregate_ragas() -> pd.DataFrame:
    path = OUTPUT_DIR / "ragas_metrics.csv"
    if not path.exists():
        return pd.DataFrame(columns=["pipeline", "k", "faithfulness"])
    return pd.read_csv(path)


def main() -> None:
    ragas_agg = _aggregate_ragas()
    rows: list[dict] = []

    for pipeline in PIPELINE_KEYS:
        for k in K_VALUES:
            per_q = load_per_question(
                PER_QUESTION_DIR, pipelines=[pipeline], ks=[k], seed=RANDOM_SEED,
            )
            if per_q.empty:
                logger.warning("No per-question rows for %s k=%d", pipeline, k)
                continue

            if "faithfulness" in per_q.columns and per_q["faithfulness"].notna().any():
                halluc = hallucination_rate(per_q)
                indicative = False
            else:
                # Fall back to aggregate-level faithfulness (no per-row filtering).
                ragas_row = ragas_agg[
                    (ragas_agg["pipeline"] == pipeline) & (ragas_agg["k"] == k)
                ]
                if ragas_row.empty:
                    halluc = float("nan")
                else:
                    # Apply aggregate (1 - faithfulness) to all attempted answerable rows.
                    halluc = 1.0 - float(ragas_row["faithfulness"].iloc[0])
                indicative = True

            refusal = refusal_rate_on_answerable(per_q)
            rows.append({
                "pipeline": pipeline,
                "k": int(k),
                "n": int(len(per_q)),
                "hallucination_rate": round(halluc, 4) if pd.notna(halluc) else "",
                "refusal_rate_on_answerable": round(refusal, 4) if pd.notna(refusal) else "",
                "indicative": indicative,
            })

            # Mirror per-pipeline
            per_pipeline_path = pipeline_output_dir(pipeline) / "hallucination_metrics.csv"
            existing = (
                pd.read_csv(per_pipeline_path) if per_pipeline_path.exists() else pd.DataFrame()
            )
            existing = existing[~((existing.get("pipeline") == pipeline) & (existing.get("k") == k))] \
                if not existing.empty else existing
            out_df = pd.concat([existing, pd.DataFrame([rows[-1]])], ignore_index=True) \
                .sort_values("k").reset_index(drop=True)
            out_df.to_csv(per_pipeline_path, index=False)

    out = OUTPUT_DIR / "hallucination_metrics.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info("Wrote %s (%d rows)", out, len(rows))


if __name__ == "__main__":
    main()
