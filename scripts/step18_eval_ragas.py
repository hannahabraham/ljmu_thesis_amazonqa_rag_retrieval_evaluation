"""RAGAS faithfulness, context_precision, context_recall — routed through Gemini.

v5: runs at **all four k values** by default (Tables 1 and 2 both need RAGAS at
multiple k). Per-row scores are written back into the per-question JSONL so
Tables 3 (category) and 4 (length bucket) can filter RAGAS by metadata.

Examples:
  python scripts/18_eval_ragas.py                       # all k × all pipelines
  python scripts/18_eval_ragas.py --ks 5                # k=5 only
  python scripts/18_eval_ragas.py --ks 10 --pipelines dense pc
"""
from __future__ import annotations

import argparse
import logging
from itertools import product
from pathlib import Path

import pandas as pd

from config.settings import (
    K_VALUES,
    OUTPUT_DIR,
    PER_QUESTION_DIR,
    PIPELINE_KEYS,
    RANDOM_SEED,
    pipeline_output_dir,
)
from src.evaluation.ragas_metrics import run_ragas
from src.utils.io import parse_list_field, read_jsonl, write_jsonl
from src.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def _load_per_question(pipeline: str, k: int, seed: int) -> pd.DataFrame | None:
    """Prefer per-question JSONL; fall back to per-pipeline answers CSV."""
    jsonl_path = PER_QUESTION_DIR / f"{pipeline}_k{k}_seed{seed}.jsonl"
    if jsonl_path.exists():
        rows = read_jsonl(jsonl_path)
        if rows:
            return pd.DataFrame(rows)
    csv_path = pipeline_output_dir(pipeline) / f"answers_k{k}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "retrieved_context" in df.columns:
            df["retrieved_context"] = df["retrieved_context"].apply(parse_list_field)
        if "retrieved_doc_ids" in df.columns:
            df["retrieved_doc_ids"] = df["retrieved_doc_ids"].apply(parse_list_field)
        return df
    logger.warning("No per-question data for %s k=%d (looked at %s and %s)",
                   pipeline, k, jsonl_path, csv_path)
    return None


def _write_back_per_row_scores(
    pipeline: str, k: int, seed: int, scores_df: pd.DataFrame,
) -> None:
    """Persist per-row faithfulness / context_precision / context_recall to JSONL."""
    jsonl_path = PER_QUESTION_DIR / f"{pipeline}_k{k}_seed{seed}.jsonl"
    if not jsonl_path.exists():
        logger.info("Skipping per-row write-back (no JSONL at %s)", jsonl_path)
        return
    rows = read_jsonl(jsonl_path)
    if len(rows) != len(scores_df):
        logger.warning(
            "Length mismatch writing per-row scores (%d JSONL vs %d RAGAS) for %s k=%d",
            len(rows), len(scores_df), pipeline, k,
        )
        return
    metric_cols = [c for c in ("faithfulness", "context_precision", "context_recall")
                   if c in scores_df.columns]
    for row, (_, score_row) in zip(rows, scores_df.iterrows()):
        for col in metric_cols:
            value = score_row[col]
            row[col] = float(value) if pd.notna(value) else None
    write_jsonl(rows, jsonl_path)
    logger.info("Updated %s with %d per-row RAGAS scores", jsonl_path, len(rows))


def _evaluate(pipeline: str, k: int, seed: int) -> dict | None:
    df = _load_per_question(pipeline, k, seed)
    if df is None or df.empty:
        return None

    if "retrieved_context" in df.columns:
        contexts = df["retrieved_context"].apply(
            lambda v: v if isinstance(v, list) else parse_list_field(v)
        )
    else:
        contexts = pd.Series([[] for _ in range(len(df))])

    ragas_input = pd.DataFrame({
        "question": df["question"],
        "answer": df["generated_answer"].fillna(""),
        "contexts": contexts,
        "ground_truth": df["gold_answer"],
    })

    result = run_ragas(ragas_input)
    scores = result.to_pandas() if hasattr(result, "to_pandas") else pd.DataFrame(result)
    out_row = {
        "pipeline": pipeline,
        "k": k,
        "n": len(ragas_input),
        "faithfulness": float(scores["faithfulness"].mean())
            if "faithfulness" in scores.columns else float("nan"),
        "context_precision": float(scores["context_precision"].mean())
            if "context_precision" in scores.columns else float("nan"),
        "context_recall": float(scores["context_recall"].mean())
            if "context_recall" in scores.columns else float("nan"),
    }

    _write_back_per_row_scores(pipeline, k, seed, scores)

    # Mirror per-pipeline
    per_pipeline_path = pipeline_output_dir(pipeline) / "ragas_metrics.csv"
    existing = (
        pd.read_csv(per_pipeline_path) if per_pipeline_path.exists() else pd.DataFrame()
    )
    existing = existing[~((existing.get("pipeline") == pipeline) & (existing.get("k") == k))] \
        if not existing.empty else existing
    out_df = pd.concat([existing, pd.DataFrame([out_row])], ignore_index=True) \
        .sort_values("k").reset_index(drop=True)
    out_df.to_csv(per_pipeline_path, index=False)
    return out_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ks", nargs="+", type=int, default=None,
        help="k values to evaluate (default: all 4 k); takes precedence over --k5-only",
    )
    parser.add_argument(
        "--k5-only", action="store_true",
        help="Evaluate only k=5 (ignored when --ks is given)",
    )
    parser.add_argument(
        "--pipelines", nargs="+", default=None,
        choices=list(PIPELINE_KEYS),
        help="pipelines to evaluate (default: all)",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    if args.ks:
        ks = args.ks
    elif args.k5_only:
        ks = [5]
    else:
        ks = list(K_VALUES)
    pipelines = args.pipelines or list(PIPELINE_KEYS)
    combos = list(product(pipelines, ks))
    logger.info("Evaluating %d (pipeline, k) cells: %s", len(combos), combos)

    rows: list[dict] = []
    for pipeline, k in combos:
        out = _evaluate(pipeline, k, seed=args.seed)
        if out is not None:
            rows.append(out)

    out_path = OUTPUT_DIR / "ragas_metrics.csv"
    new_df = pd.DataFrame(rows)
    if out_path.exists() and not new_df.empty:
        existing = pd.read_csv(out_path)
        keys = set(zip(new_df["pipeline"], new_df["k"]))
        existing = existing[~existing.apply(lambda r: (r["pipeline"], r["k"]) in keys, axis=1)]
        merged = pd.concat([existing, new_df], ignore_index=True)
    else:
        merged = new_df
    merged = merged.sort_values(["pipeline", "k"]).reset_index(drop=True)
    merged.to_csv(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(merged))


if __name__ == "__main__":
    main()
