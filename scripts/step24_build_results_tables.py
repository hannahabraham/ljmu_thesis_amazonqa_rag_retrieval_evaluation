"""Assemble the six Results Sheet tables (v5).

Reads per-question JSONL from outputs/per_question/, plus the RAGAS aggregate
and retrieval-metrics CSVs, then calls each builder in src/evaluation/table_builders.
The Table 7 builder is run again here as the canonical source of the `Rank`
column for Table 1.

Outputs to outputs/tables/:
  table1_overall.csv          5 rows
  table2_depth.csv            20 rows (5 pipelines × 4 k)
  table3_category.csv         20 rows (4 named cats × 5 pipelines, k=5)
  table4_length.csv           15 rows (3 buckets × 5 pipelines, k=5)
  table6_answerability.csv    5 rows (k=5)
  table7_final_ranking.csv    5 rows (composite over best-k)

Run scripts/22_final_ranking.py afterwards to also emit pairwise_wilcoxon.csv.
"""
from __future__ import annotations

import logging

import pandas as pd

from config.settings import (
    OUTPUT_DIR,
    PER_QUESTION_DIR,
    PIPELINE_KEYS,
    RANDOM_SEED,
    TABLES_DIR,
)
from src.evaluation.table_builders import (
    build_table1_overall,
    build_table2_depth,
    build_table3_category,
    build_table4_length,
    build_table6_answerability,
    build_table7_final_ranking,
)
from src.utils.io import load_per_question
from src.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def _load_ragas() -> pd.DataFrame:
    path = OUTPUT_DIR / "ragas_metrics.csv"
    if not path.exists():
        logger.warning("No RAGAS metrics at %s — Faithfulness/CP/CR will be blank", path)
        return pd.DataFrame(columns=["pipeline", "k", "faithfulness",
                                     "context_precision", "context_recall"])
    return pd.read_csv(path)


def _load_retrieval() -> pd.DataFrame:
    path = OUTPUT_DIR / "retrieval_metrics.csv"
    if not path.exists():
        logger.warning("No retrieval metrics at %s — Table 2 Recall@K/MRR will be blank", path)
        return pd.DataFrame(columns=["pipeline", "k", "recall_at_k", "mrr"])
    return pd.read_csv(path)


def _ensure_per_row_ragas(per_q: pd.DataFrame, ragas: pd.DataFrame) -> pd.DataFrame:
    """Broadcast aggregate RAGAS metrics onto per_q where row-level missing."""
    needed = ("faithfulness", "context_precision", "context_recall")
    for col in needed:
        if col not in per_q.columns:
            per_q[col] = pd.NA
    if ragas.empty:
        return per_q
    agg = ragas.rename(columns={c: f"_agg_{c}" for c in needed})[
        ["pipeline", "k"] + [f"_agg_{c}" for c in needed]
    ]
    df = per_q.merge(agg, on=["pipeline", "k"], how="left")
    for col in needed:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[f"_agg_{col}"])
    return df.drop(columns=[f"_agg_{c}" for c in needed if f"_agg_{c}" in df.columns])


def main() -> None:
    per_q = load_per_question(
        PER_QUESTION_DIR, pipelines=list(PIPELINE_KEYS), seed=RANDOM_SEED,
    )
    if per_q.empty:
        raise SystemExit(
            f"No per-question JSONL in {PER_QUESTION_DIR}. Run scripts/11_run_*.py first."
        )

    ragas = _load_ragas()
    retrieval = _load_retrieval()
    per_q = _ensure_per_row_ragas(per_q, ragas)

    # Table 7 first so we can backfill the Rank column on Table 1.
    table7 = build_table7_final_ranking(per_q, ragas_df=ragas)
    table7_path = TABLES_DIR / "table7_final_ranking.csv"
    table7.to_csv(table7_path, index=False)
    logger.info("Wrote %s (%d rows)", table7_path, len(table7))

    rank_lookup = dict(zip(table7["Pipeline"], table7["Rank"]))

    table1 = build_table1_overall(per_q, ragas_df=ragas)
    table1["Rank"] = table1["Architecture / Method"].map(rank_lookup).fillna("")
    table1_path = TABLES_DIR / "table1_overall.csv"
    table1.to_csv(table1_path, index=False)
    logger.info("Wrote %s (%d rows)", table1_path, len(table1))

    table2 = build_table2_depth(per_q, retrieval_df=retrieval, ragas_df=ragas)
    table2_path = TABLES_DIR / "table2_depth.csv"
    table2.to_csv(table2_path, index=False)
    logger.info("Wrote %s (%d rows)", table2_path, len(table2))

    table3 = build_table3_category(per_q, ragas_df=ragas)
    table3_path = TABLES_DIR / "table3_category.csv"
    table3.to_csv(table3_path, index=False)
    logger.info("Wrote %s (%d rows)", table3_path, len(table3))

    table4 = build_table4_length(per_q, ragas_df=ragas)
    table4_path = TABLES_DIR / "table4_length.csv"
    table4.to_csv(table4_path, index=False)
    logger.info("Wrote %s (%d rows)", table4_path, len(table4))

    table6 = build_table6_answerability(per_q, ragas_df=ragas)
    table6_path = TABLES_DIR / "table6_answerability.csv"
    table6.to_csv(table6_path, index=False)
    logger.info("Wrote %s (%d rows)", table6_path, len(table6))


if __name__ == "__main__":
    main()
