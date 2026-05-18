"""Builders for the six Results Sheet tables (v5).

Each builder accepts a per-question DataFrame (the source of truth, derived
from ``outputs/per_question/*.jsonl`` or per-pipeline answers CSVs) and an
optional aggregate RAGAS DataFrame keyed by (pipeline, k), and returns a
DataFrame whose **column names exactly match the Results Sheet**.

The per-question DataFrame is expected to contain these columns (extras OK):

    pipeline, k, golden_id, record_id, asin, category, q_bucket, question_type,
    is_answerable, refused, gold_answer, generated_answer, evidence_doc_id,
    retrieved_doc_ids (list), retrieved_context (list),
    em, token_f1, retrieval_ms, generation_ms, total_ms,
    is_correct                          (optional — recomputed if missing)

Optional per-row RAGAS columns (preferred when present, otherwise the aggregate
is broadcast):

    faithfulness, context_precision, context_recall

The aggregate ragas DataFrame, if supplied, has columns
``pipeline, k, faithfulness, context_precision, context_recall`` keyed at the
pipeline × k level.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from config.settings import (
    CORRECT_F1_THRESHOLD,
    NAMED_CATEGORIES,
    QUESTION_LENGTH_BUCKETS,
)
from src.evaluation.answerability import compute_answerability_table
from src.evaluation.generation_metrics import correct_answers_count
from src.evaluation.hallucination import (
    hallucination_rate,
    refusal_rate_on_answerable,
)
from src.evaluation.latency import avg_latency_per_question_ms

PIPELINE_LABEL = {
    "bm25": "BM25 Retrieval",
    "dense": "Dense Retrieval",
    "sentwin": "Sentence Window Retrieval",
    "hybrid": "Hybrid Retrieval",
    "pc": "Parent-Child Retrieval",
}

# Experiment-ID stem per pipeline; suffix is added per table.
PIPELINE_EXP_STEM = {
    "bm25": "BM25",
    "dense": "DENSE",
    "sentwin": "SENTWIN",
    "hybrid": "HYBRID",
    "pc": "PARENT_CHILD",
}

CATEGORY_EXP_PREFIX = {
    "Electronics": "EXP_10",
    "Toys_and_Games": "EXP_11",
    "Health_and_Personal_Care": "EXP_12",
    "Home_and_Kitchen": "EXP_13",
}

BUCKET_EXP_PREFIX = {
    "short": "EXP_15",
    "medium": "EXP_16",
    "long": "EXP_17",
}

# Baseline retrieval depth for Tables 1, 3, 4, 6 and the pairwise Wilcoxon test.
# Table 2 (depth analysis) intentionally sweeps every k; everything else reports
# the k=5 baseline so the comparison is one-shot per pipeline / category / bucket.
# See CLAUDE.md §9 (table mapping) and SAMPLE_QUOTAS in config/settings.py.
TABLE_BASELINE_K = 5


def _filter_baseline_k(df: pd.DataFrame, baseline_k: int = TABLE_BASELINE_K) -> pd.DataFrame:
    """Return rows whose ``k`` column equals the baseline depth."""
    if df.empty or "k" not in df.columns:
        return df
    return df[pd.to_numeric(df["k"], errors="coerce") == baseline_k].copy()

BUCKET_LABEL = {"short": "Short", "medium": "Medium", "long": "Long"}


def _word_count_rule() -> dict[str, str]:
    short_max, medium_max = QUESTION_LENGTH_BUCKETS
    return {
        "short": f"1–{short_max} words",
        "medium": f"{short_max + 1}–{medium_max} words",
        "long": f"{medium_max + 1}+ words",
    }


# ---------------------------------------------------------------------------
# Helpers


def _ensure_bool(series: pd.Series) -> pd.Series:
    def _coerce(v: object) -> bool:
        if v is None:
            return False
        if isinstance(v, float) and np.isnan(v):
            return False
        if isinstance(v, str):
            return v.strip().lower() in {"true", "1", "yes", "y", "t"}
        return bool(v)

    return series.apply(_coerce)


def _filter_k(per_q: pd.DataFrame, k: int) -> pd.DataFrame:
    return per_q[pd.to_numeric(per_q["k"], errors="coerce") == k].copy()


def _attach_aggregate_ragas(
    per_q: pd.DataFrame,
    ragas_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Broadcast aggregate RAGAS metrics onto per_q where row-level missing."""
    df = per_q.copy()
    needed = ("faithfulness", "context_precision", "context_recall")

    if ragas_df is None or ragas_df.empty:
        for col in needed:
            if col not in df.columns:
                df[col] = np.nan
        return df

    agg = ragas_df.rename(
        columns={c: f"_agg_{c}" for c in needed if c in ragas_df.columns}
    )[["pipeline", "k"] + [f"_agg_{c}" for c in needed if c in ragas_df.columns]]
    df = df.merge(agg, on=["pipeline", "k"], how="left")
    for col in needed:
        agg_col = f"_agg_{col}"
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce").fillna(df.get(agg_col, np.nan))
                if agg_col in df.columns
                else df[col]
            )
        elif agg_col in df.columns:
            df[col] = df[agg_col]
    return df.drop(columns=[c for c in df.columns if c.startswith("_agg_")])


def _mean_or_nan(values: pd.Series) -> float:
    coerced = pd.to_numeric(values, errors="coerce").dropna()
    if coerced.empty:
        return float("nan")
    return float(coerced.mean())


def _answerable_subset(per_q: pd.DataFrame) -> pd.DataFrame:
    if "is_answerable" not in per_q.columns:
        return per_q.iloc[0:0]
    return per_q[_ensure_bool(per_q["is_answerable"])]


def _ensure_is_correct(per_q: pd.DataFrame, threshold: float) -> pd.DataFrame:
    from src.evaluation.generation_metrics import is_correct

    df = per_q.copy()
    if "is_correct" in df.columns and df["is_correct"].notna().all():
        return df
    df["is_correct"] = df.apply(lambda r: is_correct(r, threshold), axis=1)
    return df


def _pipeline_order(values: Iterable[str]) -> list[str]:
    """Stable ordering: known labels first (in PIPELINE_LABEL order), then extras."""
    known = [p for p in PIPELINE_LABEL if p in set(values)]
    extras = sorted(set(values) - set(known))
    return known + extras


# ---------------------------------------------------------------------------
# Table 1 — Overall Pipeline Performance (all k)


TABLE1_COLUMNS = [
    "Architecture / Method",
    "K Value",
    "Experiment ID",
    "Total Questions",
    "Correct Answers",
    "Exact Match Accuracy (%)",
    "F1 Score",
    "Faithfulness Score",
    "Context Precision",
    "Context Recall",
    "Answerability Accuracy",
    "Avg Latency / Question",
    "Rank",
]


def build_table1_overall(
    per_q: pd.DataFrame,
    ragas_df: pd.DataFrame | None = None,
    f1_threshold: float = CORRECT_F1_THRESHOLD,
) -> pd.DataFrame:
    """Table 1 — one row per pipeline at the k=5 baseline (see ``TABLE_BASELINE_K``)."""
    df = per_q.copy()
    df = _attach_aggregate_ragas(df, ragas_df)
    df = _ensure_is_correct(df, f1_threshold)
    df["is_answerable"] = _ensure_bool(df["is_answerable"])
    df["refused"] = _ensure_bool(df["refused"])
    df = _filter_baseline_k(df)

    rows: list[dict] = []
    for idx, pipeline in enumerate(_pipeline_order(df["pipeline"].unique()), start=1):
        sub = df[df["pipeline"] == pipeline]
        if sub.empty:
            continue
        k = TABLE_BASELINE_K
        answerable = _answerable_subset(sub)
        ans_table = compute_answerability_table(
            sub.assign(is_answerable=sub["is_answerable"].astype(int))
        )
        rows.append(
            {
                "Architecture / Method": PIPELINE_LABEL.get(pipeline, pipeline),
                "K Value": int(k),
                "Experiment ID": (
                    f"EXP_{idx:02d}_"
                    f"{PIPELINE_EXP_STEM.get(pipeline, pipeline.upper())}_K{int(k)}"
                ),
                "Total Questions": int(len(sub)),
                "Correct Answers": correct_answers_count(sub, f1_threshold),
                "Exact Match Accuracy (%)": round(
                    _mean_or_nan(answerable.get("em", pd.Series(dtype=float))) * 100.0,
                    2,
                ),
                "F1 Score": round(
                    _mean_or_nan(answerable.get("token_f1", pd.Series(dtype=float))),
                    4,
                ),
                "Faithfulness Score": round(
                    _mean_or_nan(sub.get("faithfulness", pd.Series(dtype=float))), 4
                ),
                "Context Precision": round(
                    _mean_or_nan(sub.get("context_precision", pd.Series(dtype=float))),
                    4,
                ),
                "Context Recall": round(
                    _mean_or_nan(sub.get("context_recall", pd.Series(dtype=float))),
                    4,
                ),
                "Answerability Accuracy": round(
                    float(ans_table["answerability_acc"].iloc[0]), 4
                ),
                "Avg Latency / Question": round(
                    avg_latency_per_question_ms(sub), 2
                ),
                "Rank": "",  # filled by table7 for each pipeline label
            }
        )
    return pd.DataFrame(rows, columns=TABLE1_COLUMNS)


# ---------------------------------------------------------------------------
# Table 2 — Retrieval Depth (5 × 4 k)


TABLE2_COLUMNS = [
    "Pipeline",
    "K Value",
    "Total Questions",
    "Recall@K",
    "MRR",
    "Context Precision",
    "Context Recall",
    "F1 Score",
    "Faithfulness Score",
    "Avg Latency / Question",
]


def build_table2_depth(
    per_q: pd.DataFrame,
    retrieval_df: pd.DataFrame,
    ragas_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Table 2 — one row per (pipeline, k). retrieval_df: pipeline,k,recall_at_k,mrr."""
    df = _attach_aggregate_ragas(per_q, ragas_df).copy()
    df["is_answerable"] = _ensure_bool(df["is_answerable"])
    df["refused"] = _ensure_bool(df["refused"])

    rows: list[dict] = []
    for pipeline in _pipeline_order(df["pipeline"].unique()):
        for k in sorted(pd.to_numeric(df["k"], errors="coerce").dropna().unique()):
            sub = df[(df["pipeline"] == pipeline) & (df["k"] == k)]
            if sub.empty:
                continue
            ret = retrieval_df[
                (retrieval_df["pipeline"] == pipeline) & (retrieval_df["k"] == k)
            ]
            recall = (
                float(ret["recall_at_k"].iloc[0]) if not ret.empty else float("nan")
            )
            mrr = float(ret["mrr"].iloc[0]) if not ret.empty else float("nan")
            answerable = _answerable_subset(sub)
            rows.append(
                {
                    "Pipeline": PIPELINE_LABEL.get(pipeline, pipeline),
                    "K Value": int(k),
                    "Total Questions": int(len(sub)),
                    "Recall@K": round(recall, 4) if not np.isnan(recall) else "",
                    "MRR": round(mrr, 4) if not np.isnan(mrr) else "",
                    "Context Precision": round(
                        _mean_or_nan(
                            sub.get("context_precision", pd.Series(dtype=float))
                        ),
                        4,
                    ),
                    "Context Recall": round(
                        _mean_or_nan(sub.get("context_recall", pd.Series(dtype=float))),
                        4,
                    ),
                    "F1 Score": round(
                        _mean_or_nan(
                            answerable.get("token_f1", pd.Series(dtype=float))
                        ),
                        4,
                    ),
                    "Faithfulness Score": round(
                        _mean_or_nan(sub.get("faithfulness", pd.Series(dtype=float))), 4
                    ),
                    "Avg Latency / Question": round(
                        avg_latency_per_question_ms(sub), 2
                    ),
                }
            )
    return pd.DataFrame(rows, columns=TABLE2_COLUMNS)


# ---------------------------------------------------------------------------
# Table 3 — Category-Level (4 named cats × 5 pipelines × all k)


TABLE3_COLUMNS = [
    "Product Category",
    "Pipeline",
    "K Value",
    "Experiment ID",
    "Total Questions",
    "Exact Match Accuracy (%)",
    "F1 Score",
    "Faithfulness Score",
    "Context Precision",
    "Context Recall",
    "Answerability Accuracy",
    "Observation",
]


def build_table3_category(
    per_q: pd.DataFrame,
    ragas_df: pd.DataFrame | None = None,
    named_categories: tuple[str, ...] = NAMED_CATEGORIES,
) -> pd.DataFrame:
    """Table 3 — one row per (named_category, pipeline) at the k=5 baseline."""
    df = per_q.copy()
    df = df[df["category"].isin(named_categories)].copy()
    df = _attach_aggregate_ragas(df, ragas_df)
    df["is_answerable"] = _ensure_bool(df["is_answerable"])
    df = _filter_baseline_k(df)
    k = TABLE_BASELINE_K

    rows: list[dict] = []
    for category in named_categories:
        cat_df = df[df["category"] == category]
        if cat_df.empty:
            continue
        f1_by_pipeline: dict[str, float] = {}
        per_pipeline_rows: list[dict] = []
        for pipeline in _pipeline_order(cat_df["pipeline"].unique()):
            sub = cat_df[cat_df["pipeline"] == pipeline]
            if sub.empty:
                continue
            answerable = _answerable_subset(sub)
            f1_val = _mean_or_nan(answerable.get("token_f1", pd.Series(dtype=float)))
            f1_by_pipeline[pipeline] = f1_val
            ans_table = compute_answerability_table(
                sub.assign(is_answerable=sub["is_answerable"].astype(int))
            )
            exp_id = (
                f"{CATEGORY_EXP_PREFIX.get(category, 'EXP_XX')}_"
                f"{PIPELINE_EXP_STEM.get(pipeline, pipeline.upper())}_K{int(k)}"
            )
            per_pipeline_rows.append(
                {
                    "Product Category": category,
                    "Pipeline": PIPELINE_LABEL.get(pipeline, pipeline),
                    "K Value": int(k),
                    "Experiment ID": exp_id,
                    "Total Questions": int(len(sub)),
                    "Exact Match Accuracy (%)": round(
                        _mean_or_nan(answerable.get("em", pd.Series(dtype=float)))
                        * 100.0,
                        2,
                    ),
                    "F1 Score": round(f1_val, 4) if not np.isnan(f1_val) else "",
                    "Faithfulness Score": round(
                        _mean_or_nan(sub.get("faithfulness", pd.Series(dtype=float))),
                        4,
                    ),
                    "Context Precision": round(
                        _mean_or_nan(
                            sub.get("context_precision", pd.Series(dtype=float))
                        ),
                        4,
                    ),
                    "Context Recall": round(
                        _mean_or_nan(sub.get("context_recall", pd.Series(dtype=float))),
                        4,
                    ),
                    "Answerability Accuracy": round(
                        float(ans_table["answerability_acc"].iloc[0]), 4
                    ),
                }
            )

        if not per_pipeline_rows:
            continue
        best_pipeline = max(
            f1_by_pipeline,
            key=lambda p, scores=f1_by_pipeline: (
                -1.0 if np.isnan(scores[p]) else scores[p]
            ),
        )
        best_f1 = f1_by_pipeline[best_pipeline]
        observation = (
            f"Best in category at k={int(k)}: "
            f"{PIPELINE_LABEL.get(best_pipeline, best_pipeline)} "
            f"(F1={best_f1:.2f})"
        )
        for row in per_pipeline_rows:
            row["Observation"] = observation
            rows.append(row)
    return pd.DataFrame(rows, columns=TABLE3_COLUMNS)


# ---------------------------------------------------------------------------
# Table 4 — Question Length (3 buckets × 5 pipelines × all k)


TABLE4_COLUMNS = [
    "Question Length Bucket",
    "Word Count Rule",
    "Pipeline",
    "K Value",
    "Experiment ID",
    "Total Questions",
    "Exact Match Accuracy (%)",
    "F1 Score",
    "Faithfulness Score",
    "Context Precision",
    "Context Recall",
    "Observation",
]


def build_table4_length(
    per_q: pd.DataFrame,
    ragas_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Table 4 — one row per (bucket, pipeline) at the k=5 baseline."""
    df = per_q.copy()
    df = _attach_aggregate_ragas(df, ragas_df)
    df["is_answerable"] = _ensure_bool(df["is_answerable"])
    df = _filter_baseline_k(df)
    word_count_rule = _word_count_rule()
    k = TABLE_BASELINE_K

    rows: list[dict] = []
    for bucket in ("short", "medium", "long"):
        bucket_df = df[df["q_bucket"] == bucket]
        if bucket_df.empty:
            continue
        f1_by_pipeline: dict[str, float] = {}
        per_pipeline_rows: list[dict] = []
        for pipeline in _pipeline_order(bucket_df["pipeline"].unique()):
            sub = bucket_df[bucket_df["pipeline"] == pipeline]
            if sub.empty:
                continue
            answerable = _answerable_subset(sub)
            f1_val = _mean_or_nan(answerable.get("token_f1", pd.Series(dtype=float)))
            f1_by_pipeline[pipeline] = f1_val
            exp_id = (
                f"{BUCKET_EXP_PREFIX.get(bucket, 'EXP_XX')}_"
                f"{PIPELINE_EXP_STEM.get(pipeline, pipeline.upper())}_K{int(k)}"
            )
            per_pipeline_rows.append(
                {
                    "Question Length Bucket": BUCKET_LABEL[bucket],
                    "Word Count Rule": word_count_rule[bucket],
                    "Pipeline": PIPELINE_LABEL.get(pipeline, pipeline),
                    "K Value": int(k),
                    "Experiment ID": exp_id,
                    "Total Questions": int(len(sub)),
                    "Exact Match Accuracy (%)": round(
                        _mean_or_nan(answerable.get("em", pd.Series(dtype=float)))
                        * 100.0,
                        2,
                    ),
                    "F1 Score": round(f1_val, 4) if not np.isnan(f1_val) else "",
                    "Faithfulness Score": round(
                        _mean_or_nan(sub.get("faithfulness", pd.Series(dtype=float))),
                        4,
                    ),
                    "Context Precision": round(
                        _mean_or_nan(
                            sub.get("context_precision", pd.Series(dtype=float))
                        ),
                        4,
                    ),
                    "Context Recall": round(
                        _mean_or_nan(sub.get("context_recall", pd.Series(dtype=float))),
                        4,
                    ),
                }
            )
        if not per_pipeline_rows:
            continue
        best_pipeline = max(
            f1_by_pipeline,
            key=lambda p, scores=f1_by_pipeline: (
                -1.0 if np.isnan(scores[p]) else scores[p]
            ),
        )
        best_f1 = f1_by_pipeline[best_pipeline]
        observation = (
            f"Best in bucket at k={int(k)}: "
            f"{PIPELINE_LABEL.get(best_pipeline, best_pipeline)} "
            f"(F1={best_f1:.2f})"
        )
        for row in per_pipeline_rows:
            row["Observation"] = observation
            rows.append(row)
    return pd.DataFrame(rows, columns=TABLE4_COLUMNS)


# ---------------------------------------------------------------------------
# Table 6 — Answerability (all k, one row per pipeline/k)


TABLE6_COLUMNS = [
    "Pipeline",
    "K Value",
    "Total Answerable",
    "Total Unanswerable",
    "Correctly Answered",
    "Wrongly Refused",
    "Wrongly Answered",
    "Correctly Refused",
    "Answerability Accuracy",
    "Hallucination Rate",
    "Refusal Rate on Answerable",
]


def build_table6_answerability(
    per_q: pd.DataFrame,
    ragas_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Table 6 — confusion table + hallucination/refusal pair at the k=5 baseline."""
    df = per_q.copy()
    df = _attach_aggregate_ragas(df, ragas_df)
    df["is_answerable"] = _ensure_bool(df["is_answerable"])
    df["refused"] = _ensure_bool(df["refused"])
    df = _filter_baseline_k(df)
    k = TABLE_BASELINE_K

    rows: list[dict] = []
    for pipeline in _pipeline_order(df["pipeline"].unique()):
        sub = df[df["pipeline"] == pipeline]
        if sub.empty:
            continue
        n_answerable = int(sub["is_answerable"].sum())
        n_unanswerable = int((~sub["is_answerable"]).sum())
        ans_input = sub.assign(is_answerable=sub["is_answerable"].astype(int))
        table = compute_answerability_table(ans_input)
        rows.append(
            {
                "Pipeline": PIPELINE_LABEL.get(pipeline, pipeline),
                "K Value": int(k),
                "Total Answerable": n_answerable,
                "Total Unanswerable": n_unanswerable,
                "Correctly Answered": int(table["correctly_answered"].iloc[0]),
                "Wrongly Refused": int(table["wrongly_refused"].iloc[0]),
                "Wrongly Answered": int(table["wrongly_answered"].iloc[0]),
                "Correctly Refused": int(table["correctly_refused"].iloc[0]),
                "Answerability Accuracy": round(
                    float(table["answerability_acc"].iloc[0]), 4
                ),
                "Hallucination Rate": round(hallucination_rate(sub), 4),
                "Refusal Rate on Answerable": round(
                    refusal_rate_on_answerable(sub), 4
                ),
            }
        )
    return pd.DataFrame(rows, columns=TABLE6_COLUMNS)


# ---------------------------------------------------------------------------
# Table 7 — Final Ranking (composite over best-k per pipeline)


TABLE7_COLUMNS = [
    "Pipeline",
    "Best K",
    "F1 @ Best K",
    "Faithfulness @ Best K",
    "Context Precision @ Best K",
    "Context Recall @ Best K",
    "Answerability Accuracy @ Best K",
    "Avg Latency @ Best K",
    "Composite Score",
    "Rank",
]

COMPOSITE_WEIGHTS = {
    "f1": 0.25,
    "faithfulness": 0.20,
    "context_precision": 0.15,
    "context_recall": 0.10,
    "answerability_acc": 0.15,
    "category_consistency": 0.10,
    "latency": 0.05,
}


def _safe(value: float | int | None) -> float:
    """Coerce None / NaN to 0.0 so composite arithmetic doesn't propagate gaps."""
    if value is None:
        return 0.0
    coerced = float(value)
    return 0.0 if np.isnan(coerced) else coerced


def _min_max_norm(values: dict[str, float]) -> dict[str, float]:
    raw = {k: _safe(v) for k, v in values.items()}
    lo = min(raw.values()) if raw else 0.0
    hi = max(raw.values()) if raw else 1.0
    if hi == lo:
        return dict.fromkeys(raw, 0.0)
    return {k: (v - lo) / (hi - lo) for k, v in raw.items()}


def _per_pipeline_at_k(
    per_q: pd.DataFrame,
    pipeline: str,
    k: int,
) -> dict[str, float]:
    sub = per_q[(per_q["pipeline"] == pipeline) & (per_q["k"] == k)].copy()
    if sub.empty:
        return {
            "f1": float("nan"),
            "faithfulness": float("nan"),
            "context_precision": float("nan"),
            "context_recall": float("nan"),
            "answerability_acc": float("nan"),
            "avg_latency_ms": float("nan"),
        }
    sub["is_answerable"] = _ensure_bool(sub["is_answerable"])
    answerable = _answerable_subset(sub)
    ans_table = compute_answerability_table(
        sub.assign(is_answerable=sub["is_answerable"].astype(int))
    )
    return {
        "f1": _mean_or_nan(answerable.get("token_f1", pd.Series(dtype=float))),
        "faithfulness": _mean_or_nan(sub.get("faithfulness", pd.Series(dtype=float))),
        "context_precision": _mean_or_nan(
            sub.get("context_precision", pd.Series(dtype=float))
        ),
        "context_recall": _mean_or_nan(
            sub.get("context_recall", pd.Series(dtype=float))
        ),
        "answerability_acc": float(ans_table["answerability_acc"].iloc[0]),
        "avg_latency_ms": avg_latency_per_question_ms(sub),
    }


def _category_consistency(
    per_q: pd.DataFrame,
    pipeline: str,
    k: int,
    named_categories: tuple[str, ...] = NAMED_CATEGORIES,
) -> float:
    """1 / (1 + std(F1 across named categories)). Bounded in (0, 1]."""
    sub = per_q[
        (per_q["pipeline"] == pipeline)
        & (per_q["k"] == k)
        & (per_q["category"].isin(named_categories))
    ].copy()
    if sub.empty:
        return float("nan")
    sub["is_answerable"] = _ensure_bool(sub["is_answerable"])
    answerable = _answerable_subset(sub)
    if answerable.empty:
        return float("nan")
    f1s = (
        answerable.groupby("category")["token_f1"]
        .apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
        .dropna()
        .tolist()
    )
    if len(f1s) < 2:
        return 1.0
    return 1.0 / (1.0 + float(np.std(f1s, ddof=0)))


def build_table7_final_ranking(
    per_q: pd.DataFrame,
    ragas_df: pd.DataFrame | None = None,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Table 7 — composite score per pipeline at best-k (argmax over k=1,3,5,10)."""
    weights = COMPOSITE_WEIGHTS if weights is None else weights
    df = _attach_aggregate_ragas(per_q, ragas_df).copy()
    pipelines = _pipeline_order(df["pipeline"].unique())
    k_values = sorted(pd.to_numeric(df["k"], errors="coerce").dropna().unique())

    best_k: dict[str, int] = {}
    per_pipeline_metrics: dict[str, dict[str, float]] = {}
    for pipeline in pipelines:
        best_score = -np.inf
        chosen_k = int(k_values[0]) if k_values else 5
        chosen_metrics: dict[str, float] = {}
        for k in k_values:
            m = _per_pipeline_at_k(df, pipeline, int(k))
            cc = _category_consistency(df, pipeline, int(k))
            crude_score = (
                _safe(m["f1"])
                + _safe(m["faithfulness"])
                + _safe(m["context_precision"])
                + _safe(m["context_recall"])
                + _safe(m["answerability_acc"])
                + _safe(cc)
            )
            if crude_score > best_score:
                best_score = crude_score
                chosen_k = int(k)
                chosen_metrics = {**m, "category_consistency": cc}
        best_k[pipeline] = chosen_k
        per_pipeline_metrics[pipeline] = chosen_metrics

    latency_norm = _min_max_norm(
        {
            p: per_pipeline_metrics[p].get("avg_latency_ms", float("nan"))
            for p in pipelines
        }
    )

    composites: dict[str, float] = {}
    for pipeline in pipelines:
        m = per_pipeline_metrics[pipeline]
        composites[pipeline] = (
            weights["f1"] * _safe(m.get("f1"))
            + weights["faithfulness"] * _safe(m.get("faithfulness"))
            + weights["context_precision"] * _safe(m.get("context_precision"))
            + weights["context_recall"] * _safe(m.get("context_recall"))
            + weights["answerability_acc"] * _safe(m.get("answerability_acc"))
            + weights["category_consistency"] * _safe(m.get("category_consistency"))
            + weights["latency"] * (1.0 - latency_norm.get(pipeline, 0.0))
        )

    ranks = pd.Series(composites).rank(method="min", ascending=False).astype(int)

    rows = []
    for pipeline in pipelines:
        m = per_pipeline_metrics[pipeline]
        rows.append(
            {
                "Pipeline": PIPELINE_LABEL.get(pipeline, pipeline),
                "Best K": best_k[pipeline],
                "F1 @ Best K": round(_safe(m.get("f1")), 4),
                "Faithfulness @ Best K": round(_safe(m.get("faithfulness")), 4),
                "Context Precision @ Best K": round(
                    _safe(m.get("context_precision")), 4
                ),
                "Context Recall @ Best K": round(_safe(m.get("context_recall")), 4),
                "Answerability Accuracy @ Best K": round(
                    _safe(m.get("answerability_acc")), 4
                ),
                "Avg Latency @ Best K": round(_safe(m.get("avg_latency_ms")), 2),
                "Composite Score": round(composites[pipeline], 4),
                "Rank": int(ranks[pipeline]),
            }
        )
    return (
        pd.DataFrame(rows, columns=TABLE7_COLUMNS)
        .sort_values("Rank")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Pairwise Wilcoxon block (companion to Table 7)


PAIRWISE_COLUMNS = [
    "K Value",
    "Pipeline A",
    "Pipeline B",
    "Metric",
    "Median Diff (A - B)",
    "Wilcoxon W",
    "p_value",
    "n_pairs",
    "Significant at α=0.05",
    "Significant after Bonferroni (α/10)",
]


def build_pairwise_wilcoxon(
    per_q: pd.DataFrame,
    metrics: tuple[str, ...] = ("token_f1", "faithfulness"),
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Pairwise Wilcoxon on per-question scores at the k=5 baseline, one row per pair × metric.

    Pairs rows on ``record_id`` so the test is properly paired across pipelines.
    Bonferroni correction is reported alongside the raw p-value. Restricted to the
    k=5 baseline so the pair counts match the rest of the Results Sheet — Table 2
    is the place for the depth sweep.
    """
    from src.evaluation.statistics import bonferroni_threshold, paired_wilcoxon

    df = per_q.copy()
    if df.empty:
        return pd.DataFrame(columns=PAIRWISE_COLUMNS)

    df = _filter_baseline_k(df)
    if df.empty:
        return pd.DataFrame(columns=PAIRWISE_COLUMNS)

    pipelines = _pipeline_order(df["pipeline"].unique())
    pair_count = len(pipelines) * (len(pipelines) - 1) // 2
    bonf = bonferroni_threshold(alpha, max(pair_count, 1))
    k = TABLE_BASELINE_K

    rows: list[dict] = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        wide = df.pivot_table(
            index="record_id",
            columns="pipeline",
            values=metric,
            aggfunc="mean",
        ).dropna(how="any")
        for i, a in enumerate(pipelines):
            for b in pipelines[i + 1 :]:
                if a not in wide.columns or b not in wide.columns:
                    continue
                series_a = wide[a].astype(float).tolist()
                series_b = wide[b].astype(float).tolist()
                result = paired_wilcoxon(series_a, series_b)
                p = result["p_value"]
                rows.append(
                    {
                        "K Value": int(k),
                        "Pipeline A": PIPELINE_LABEL.get(a, a),
                        "Pipeline B": PIPELINE_LABEL.get(b, b),
                        "Metric": metric,
                        "Median Diff (A - B)": (
                            round(result["median_diff"], 4)
                            if result["median_diff"] is not None
                            else ""
                        ),
                        "Wilcoxon W": (
                            result["statistic"]
                            if result["statistic"] is not None
                            else ""
                        ),
                        "p_value": p if p is not None else "",
                        "n_pairs": int(result["n_pairs"] or 0),
                        "Significant at α=0.05": (p is not None and p < alpha),
                        "Significant after Bonferroni (α/10)": (
                            p is not None and p < bonf
                        ),
                    }
                )
    return pd.DataFrame(rows, columns=PAIRWISE_COLUMNS)
