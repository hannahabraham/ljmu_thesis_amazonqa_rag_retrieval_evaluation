"""Confidence intervals for metric reporting.

Bootstrap for continuous metrics (F1, ROUGE-L, faithfulness, latency).
Wilson score for proportions (accuracy, hit-rate, answerability).
"""
from __future__ import annotations

import numpy as np
from statsmodels.stats.proportion import proportion_confint

from config.settings import INDICATIVE_THRESHOLD


def bootstrap_ci(
    values: list[float],
    confidence: float = 0.95,
    n_resamples: int = 1000,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Return (mean, lower_bound, upper_bound) using percentile bootstrap."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(random_state)
    arr = np.asarray(values, dtype=float)
    means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_resamples)
    ])
    alpha = 1.0 - confidence
    lower, upper = np.quantile(means, [alpha / 2, 1.0 - alpha / 2])
    return float(arr.mean()), float(lower), float(upper)


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """Wilson score CI for a binomial proportion."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    point = successes / n
    lower, upper = proportion_confint(successes, n, alpha=1.0 - confidence, method="wilson")
    return point, float(lower), float(upper)


def is_indicative(n: int) -> bool:
    return n < INDICATIVE_THRESHOLD


def paired_wilcoxon(
    scores_a: list[float], scores_b: list[float],
) -> dict[str, float | int | None]:
    """Paired Wilcoxon signed-rank for "A vs B on a per-question metric".

    Same question order across A and B is assumed (e.g. per-record token_f1).
    Returns statistic, p-value, count of non-zero diffs, and the median diff
    (A − B). Returns None values when there are <6 non-zero pairs (scipy
    minimum for a meaningful test).
    """
    from scipy.stats import wilcoxon

    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"paired Wilcoxon requires equal-length series ({len(scores_a)} vs {len(scores_b)})"
        )
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    nonzero = [d for d in diffs if d != 0.0]
    if len(nonzero) < 6:
        return {
            "statistic": None,
            "p_value": None,
            "n_pairs": len(nonzero),
            "median_diff": float(np.median(diffs)) if diffs else None,
        }
    statistic, p_value = wilcoxon(scores_a, scores_b)
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n_pairs": len(nonzero),
        "median_diff": float(np.median(diffs)),
    }


def bonferroni_threshold(alpha: float, n_tests: int) -> float:
    """Bonferroni-adjusted significance threshold (α / n_tests)."""
    if n_tests <= 0:
        return alpha
    return alpha / n_tests
