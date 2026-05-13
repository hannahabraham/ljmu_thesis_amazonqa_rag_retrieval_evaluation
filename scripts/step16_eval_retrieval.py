"""Aggregate retrieval metrics across all (pipeline, k) results.

Reads outputs/<pipeline>/answers_k<k>.csv and reports Hit@K, Recall@K, MRR, and
nDCG@K with bootstrap 95% CIs. Skips rows with no golden evidence_doc_id.
"""
from __future__ import annotations

import logging
from itertools import product

import pandas as pd

from config.settings import K_VALUES, OUTPUT_DIR, PIPELINE_KEYS, pipeline_output_dir
from src.evaluation.retrieval_metrics import (
    hit_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.evaluation.statistics import bootstrap_ci
from src.utils.io import parse_list_field
from src.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    rows: list[dict] = []
    for pipeline, k in product(PIPELINE_KEYS, K_VALUES):
        path = pipeline_output_dir(pipeline) / f"answers_k{k}.csv"
        if not path.exists():
            logger.warning("Missing %s, skipping", path)
            continue
        df = pd.read_csv(path)
        df = df[df["evidence_doc_id"].notna()]
        if df.empty:
            continue

        df = df.copy()
        df["retrieved_doc_ids"] = df["retrieved_doc_ids"].apply(parse_list_field)

        hits = [hit_at_k(r["retrieved_doc_ids"], r["evidence_doc_id"], k) for _, r in df.iterrows()]
        recs = [recall_at_k(r["retrieved_doc_ids"], r["evidence_doc_id"], k) for _, r in df.iterrows()]
        rrs = [reciprocal_rank(r["retrieved_doc_ids"], r["evidence_doc_id"]) for _, r in df.iterrows()]
        ndcgs = [ndcg_at_k(r["retrieved_doc_ids"], r["evidence_doc_id"], k) for _, r in df.iterrows()]

        hit_mean, hit_lo, hit_hi = bootstrap_ci([float(h) for h in hits])
        rec_mean, rec_lo, rec_hi = bootstrap_ci([float(r) for r in recs])
        mrr_mean, mrr_lo, mrr_hi = bootstrap_ci([float(r) for r in rrs])
        ndcg_mean, ndcg_lo, ndcg_hi = bootstrap_ci([float(r) for r in ndcgs])

        row = {
            "pipeline": pipeline,
            "k": k,
            "n": len(df),
            "hit_at_k": hit_mean, "hit_at_k_lo": hit_lo, "hit_at_k_hi": hit_hi,
            "recall_at_k": rec_mean, "recall_at_k_lo": rec_lo, "recall_at_k_hi": rec_hi,
            "mrr": mrr_mean, "mrr_lo": mrr_lo, "mrr_hi": mrr_hi,
            "ndcg_at_k": ndcg_mean, "ndcg_at_k_lo": ndcg_lo, "ndcg_at_k_hi": ndcg_hi,
        }
        rows.append(row)

        # Per-pipeline copy of the same row goes alongside the cell artefacts
        per_pipeline_path = pipeline_output_dir(pipeline) / "retrieval_metrics.csv"
        existing = (
            pd.read_csv(per_pipeline_path) if per_pipeline_path.exists()
            else pd.DataFrame()
        )
        existing = existing[~((existing.get("pipeline") == pipeline) & (existing.get("k") == k))] \
            if not existing.empty else existing
        out_df = pd.concat([existing, pd.DataFrame([row])], ignore_index=True) \
            .sort_values("k").reset_index(drop=True)
        out_df.to_csv(per_pipeline_path, index=False)

    out = OUTPUT_DIR / "retrieval_metrics.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
