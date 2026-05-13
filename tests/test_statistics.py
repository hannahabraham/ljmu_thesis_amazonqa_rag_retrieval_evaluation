"""Unit tests for statistical evaluation utilities."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.evaluation.statistics import (
    bonferroni_threshold,
    bootstrap_ci,
    is_indicative,
    paired_wilcoxon,
    wilson_ci,
)


def test_bootstrap_ci_centered() -> None:
    """Test bootstrap confidence interval contains the estimated mean."""
    values = list(np.random.default_rng(0).normal(0.5, 0.1, 200))

    mean, lower_bound, upper_bound = bootstrap_ci(values, n_resamples=500)

    assert lower_bound < mean < upper_bound
    assert abs(mean - 0.5) < 0.05


def test_bootstrap_ci_empty() -> None:
    """Test bootstrap confidence interval returns NaN values when empty."""
    mean, lower_bound, upper_bound = bootstrap_ci([])

    assert all(math.isnan(value) for value in (mean, lower_bound, upper_bound))


def test_wilson_ci_proportion() -> None:
    """Test Wilson confidence interval for a valid proportion."""
    point, lower_bound, upper_bound = wilson_ci(8, 10)

    assert 0 <= lower_bound <= point <= upper_bound <= 1
    assert point == 0.8


def test_wilson_ci_zero_sample() -> None:
    """Test Wilson confidence interval returns NaN values for zero samples."""
    point, lower_bound, upper_bound = wilson_ci(0, 0)

    assert all(math.isnan(value) for value in (point, lower_bound, upper_bound))


def test_indicative_threshold() -> None:
    """Test indicative threshold logic."""
    assert is_indicative(5) is True
    assert is_indicative(9) is True
    assert is_indicative(10) is False
    assert is_indicative(100) is False


def test_paired_wilcoxon_detects_difference() -> None:
    """Test paired Wilcoxon detects a consistent paired difference."""
    rng = np.random.default_rng(0)
    first_values = rng.normal(0.7, 0.1, 50).tolist()
    second_values = [value - 0.15 for value in first_values]

    result = paired_wilcoxon(first_values, second_values)

    assert result["statistic"] is not None
    assert result["p_value"] < 0.05
    assert result["n_pairs"] == 50
    assert result["median_diff"] > 0


def test_paired_wilcoxon_rejects_unequal_length() -> None:
    """Test paired Wilcoxon rejects unequal-length input arrays."""
    with pytest.raises(ValueError, match="equal-length"):
        paired_wilcoxon([0.1, 0.2], [0.1])


def test_paired_wilcoxon_too_few_nonzero_pairs() -> None:
    """Test paired Wilcoxon returns null statistics for zero differences."""
    first_values = [0.5] * 10
    second_values = [0.5] * 10

    result = paired_wilcoxon(first_values, second_values)

    assert result["statistic"] is None
    assert result["p_value"] is None
    assert result["n_pairs"] == 0


def test_bonferroni_threshold() -> None:
    """Test Bonferroni threshold calculation."""
    assert abs(bonferroni_threshold(0.05, 10) - 0.005) < 1e-9
    assert bonferroni_threshold(0.05, 0) == 0.05