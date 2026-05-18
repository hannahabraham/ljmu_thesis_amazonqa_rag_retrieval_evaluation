"""Latency aggregates for the result tables (v5).

Tables 1 and 2 only show ``Avg Latency / Question`` in ms; the p50 and p95
retrieval-side numbers are written to ``outputs/latency_detail.csv`` as a
sidecar for figures and the discussion chapter.
"""
from __future__ import annotations

import pandas as pd


def _numeric_column(per_q: pd.DataFrame, column: str) -> pd.Series:
    """Return a coerced-numeric Series for ``column``, or an empty float Series if absent."""
    if column not in per_q.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(per_q[column], errors="coerce")


def avg_latency_per_question_ms(per_q: pd.DataFrame) -> float:
    """Mean total latency per question (retrieval + generation)."""
    if per_q.empty:
        return float("nan")
    if "total_ms" in per_q.columns:
        return float(_numeric_column(per_q, "total_ms").mean())
    if "retrieval_ms" in per_q.columns and "generation_ms" in per_q.columns:
        total = (
            _numeric_column(per_q, "retrieval_ms").fillna(0.0)
            + _numeric_column(per_q, "generation_ms").fillna(0.0)
        )
        return float(total.mean()) if len(total) else float("nan")
    return float("nan")


def latency_detail(per_q: pd.DataFrame) -> dict[str, float]:
    """Sidecar percentiles. Reported in `outputs/latency_detail.csv`, not tables."""
    if per_q.empty:
        return {
            "retrieval_p50_ms": float("nan"),
            "retrieval_p95_ms": float("nan"),
            "generation_p50_ms": float("nan"),
            "generation_p95_ms": float("nan"),
            "total_p50_ms": float("nan"),
            "total_p95_ms": float("nan"),
        }
    retrieval = _numeric_column(per_q, "retrieval_ms").dropna()
    generation = _numeric_column(per_q, "generation_ms").dropna()
    total = _numeric_column(per_q, "total_ms").dropna()
    if total.empty and not retrieval.empty and not generation.empty:
        total = retrieval.reset_index(drop=True) + generation.reset_index(drop=True)

    def _pct(series: pd.Series, q: float) -> float:
        return float(series.quantile(q)) if not series.empty else float("nan")

    return {
        "retrieval_p50_ms": _pct(retrieval, 0.5),
        "retrieval_p95_ms": _pct(retrieval, 0.95),
        "generation_p50_ms": _pct(generation, 0.5),
        "generation_p95_ms": _pct(generation, 0.95),
        "total_p50_ms": _pct(total, 0.5),
        "total_p95_ms": _pct(total, 0.95),
    }
