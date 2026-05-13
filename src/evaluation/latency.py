"""Latency aggregates for the result tables (v5).

Tables 1 and 2 only show ``Avg Latency / Question`` in ms; the p50 and p95
retrieval-side numbers are written to ``outputs/latency_detail.csv`` as a
sidecar for figures and the discussion chapter.
"""
from __future__ import annotations

import pandas as pd


def avg_latency_per_question_ms(per_q: pd.DataFrame) -> float:
    """Mean total latency per question (retrieval + generation)."""
    if per_q.empty or "total_ms" not in per_q.columns:
        if "retrieval_ms" in per_q.columns and "generation_ms" in per_q.columns:
            total = pd.to_numeric(per_q["retrieval_ms"], errors="coerce").fillna(0.0) + \
                pd.to_numeric(per_q["generation_ms"], errors="coerce").fillna(0.0)
            return float(total.mean()) if len(total) else float("nan")
        return float("nan")
    return float(pd.to_numeric(per_q["total_ms"], errors="coerce").mean())


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
    retrieval = pd.to_numeric(per_q.get("retrieval_ms"), errors="coerce").dropna()
    generation = pd.to_numeric(per_q.get("generation_ms"), errors="coerce").dropna()
    total = pd.to_numeric(per_q.get("total_ms"), errors="coerce").dropna()
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
