"""Robustness metrics: long-context accuracy and noise robustness.

These metrics characterise how a pipeline degrades when the retrieved context
gets long or noisy — i.e., how well the retriever and generator stay correct
when the gold evidence is buried among distractors.

Definitions used in this thesis
-------------------------------
* **Long-context accuracy**  — answerability accuracy *and* token-F1 restricted
  to questions in the `long` q_bucket (>= 13 tokens). The thesis already has
  q_bucket on every row from `05_stratified_sample.py`; this is just the slice.

* **Noise robustness**       — performance on rows where retrieval pulled in
  many irrelevant documents. Concretely, for each row we compute a "noise
  ratio" = (number of retrieved docs that don't equal evidence_doc_id) / k.
  Rows in the top quartile of noise ratio form the "noisy" subset; we report
  token-F1 and answerability accuracy on that subset, plus the *delta* vs the
  bottom-quartile (clean) subset. A small delta = the pipeline is robust to
  retrieval noise.

  In single-evidence regime this is admittedly a coarse signal — every row at
  k=5 with one gold doc has noise ratio 0.8 — so noise robustness is most
  informative at the larger k values (k=10) and across pipelines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.answerability import compute_answerability_table
from src.evaluation.generation_metrics import token_f1


def _f1_series(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [token_f1(r["generated_answer"], r["gold_answer"]) for _, r in df.iterrows()],
        index=df.index,
        dtype=float,
    )


def long_context_metrics(
    answers_df: pd.DataFrame,
    long_bucket: str = "long",
) -> dict[str, float]:
    """Token-F1 + answerability accuracy on the long q_bucket only.

    Returns NaNs (and n=0) if there are no long questions in the slice.
    Derives q_bucket from the question text if it's missing or all-NaN.
    """
    from src.sampling import assign_q_bucket

    df = answers_df.copy()
    if "q_bucket" not in df.columns or df["q_bucket"].isna().all():
        if "question" not in df.columns:
            return {"long_context_n": 0, "long_context_f1": float("nan"),
                    "long_context_answerability": float("nan")}
        df["q_bucket"] = df["question"].apply(assign_q_bucket)

    long_df = df[df["q_bucket"] == long_bucket].copy()
    if long_df.empty:
        return {"long_context_n": 0, "long_context_f1": float("nan"),
                "long_context_answerability": float("nan")}

    answerable = long_df[long_df["gold_answer"].astype(str).str.upper() != "[UNANSWERABLE]"]
    f1 = float(_f1_series(answerable).mean()) if not answerable.empty else float("nan")

    ans_eval = long_df[long_df["is_answerable"].notna()].copy()
    if not ans_eval.empty:
        ans_eval["is_answerable"] = ans_eval["is_answerable"].astype(int)
        table = compute_answerability_table(ans_eval)
        acc = float(table["answerability_acc"].iloc[0])
    else:
        acc = float("nan")

    return {
        "long_context_n": int(len(long_df)),
        "long_context_f1": f1,
        "long_context_answerability": acc,
    }


def _noise_ratios(df: pd.DataFrame, k: int) -> pd.Series:
    """Per-row fraction of retrieved docs that are not gold."""
    def _ratio(row: pd.Series) -> float:
        retrieved = row.get("retrieved_doc_ids") or []
        if not isinstance(retrieved, (list, tuple)):
            retrieved = list(retrieved) if hasattr(retrieved, "__iter__") else []
        gold = row.get("evidence_doc_id")
        topk = list(retrieved)[:k]
        if not topk:
            return float("nan")
        if gold is None or (isinstance(gold, float) and np.isnan(gold)):
            return float("nan")
        return sum(1 for d in topk if d != gold) / len(topk)
    return df.apply(_ratio, axis=1)


def noise_robustness_metrics(
    answers_df: pd.DataFrame,
    k: int,
) -> dict[str, float]:
    """F1 and answerability on the noisiest quartile of rows, plus clean-vs-noisy delta.

    Requires `retrieved_doc_ids` to already be parsed as a list (the runner does
    this before passing the frame in).
    """
    df = answers_df.copy()
    df = df[df["evidence_doc_id"].notna()]
    if df.empty:
        return {
            "noise_n": 0,
            "noise_robust_f1": float("nan"),
            "noise_robust_answerability": float("nan"),
            "noise_f1_delta": float("nan"),
        }

    df["_noise_ratio"] = _noise_ratios(df, k)
    df = df[df["_noise_ratio"].notna()]
    if df.empty or df["_noise_ratio"].nunique() < 2:
        # In single-evidence k=5 every row has the same ratio; degrade gracefully.
        f1 = float(_f1_series(
            df[df["gold_answer"].astype(str).str.upper() != "[UNANSWERABLE]"]
        ).mean()) if not df.empty else float("nan")
        return {
            "noise_n": int(len(df)),
            "noise_robust_f1": f1,
            "noise_robust_answerability": float("nan"),
            "noise_f1_delta": float("nan"),
        }

    upper = df["_noise_ratio"].quantile(0.75)
    lower = df["_noise_ratio"].quantile(0.25)
    noisy = df[df["_noise_ratio"] >= upper]
    clean = df[df["_noise_ratio"] <= lower]

    def _slice_f1(slice_df: pd.DataFrame) -> float:
        answerable = slice_df[slice_df["gold_answer"].astype(str).str.upper() != "[UNANSWERABLE]"]
        if answerable.empty:
            return float("nan")
        return float(_f1_series(answerable).mean())

    def _slice_answerability(slice_df: pd.DataFrame) -> float:
        ans_df = slice_df[slice_df["is_answerable"].notna()].copy()
        if ans_df.empty:
            return float("nan")
        ans_df["is_answerable"] = ans_df["is_answerable"].astype(int)
        return float(compute_answerability_table(ans_df)["answerability_acc"].iloc[0])

    f1_noisy = _slice_f1(noisy)
    f1_clean = _slice_f1(clean)
    delta = (f1_clean - f1_noisy) if (f1_clean == f1_clean and f1_noisy == f1_noisy) else float("nan")

    return {
        "noise_n": int(len(noisy)),
        "noise_robust_f1": f1_noisy,
        "noise_robust_answerability": _slice_answerability(noisy),
        "noise_f1_delta": delta,
    }
