"""Two-stage stratified sampling for the n=SAMPLE_SIZE evaluation set.

Stage 1 floors the four named categories (NAMED_CATEGORIES) at
MIN_PER_NAMED_CATEGORY records each, stratified by (questionType, is_answerable)
within each named category, so Table 3 cells always have ≥ the floor.

Stage 2 fills the remainder from the rest of the combined dataset, stratified
by (questionType, is_answerable). Both stages preserve the train/val/test split
proportions configured via SAMPLE_SIZE / TRAIN_SAMPLE / VAL_SAMPLE / TEST_SAMPLE.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config.settings import (
    MIN_PER_NAMED_CATEGORY,
    NAMED_CATEGORIES,
    QUESTION_LENGTH_BUCKETS,
    RANDOM_SEED,
    SAMPLE_SIZE,
    TEST_SAMPLE,
    TRAIN_SAMPLE,
    VAL_SAMPLE,
)

logger = logging.getLogger(__name__)


def assign_q_bucket(question: str) -> str:
    """short <=5, medium 6-12, long 13+ tokens."""
    short_max, medium_max = QUESTION_LENGTH_BUCKETS
    n = len(str(question).split())
    if n <= short_max:
        return "short"
    if n <= medium_max:
        return "medium"
    return "long"


def _proportional_stratified_pick(
    df: pd.DataFrame, n: int, seed: int,
) -> pd.DataFrame:
    """Sample n rows from df stratified by (questionType, is_answerable).

    Falls back to a random sample if df is smaller than n or strata are missing.
    """
    if df.empty or n <= 0:
        return df.iloc[0:0].copy()
    if len(df) <= n:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    work = df.copy()
    if "questionType" not in work.columns:
        work["questionType"] = "unknown"
    if "is_answerable" not in work.columns:
        work["is_answerable"] = -1

    work["_strata"] = (
        work["questionType"].astype(str) + "|" + work["is_answerable"].astype(str)
    )
    proportions = work["_strata"].value_counts(normalize=True)
    rng = np.random.default_rng(seed)

    target = (proportions * n).round().astype(int)
    diff = n - int(target.sum())
    if diff != 0:
        for stratum in proportions.index:
            if diff == 0:
                break
            step = 1 if diff > 0 else -1
            new_value = int(target[stratum]) + step
            if new_value >= 0:
                target[stratum] = new_value
                diff -= step

    pieces: list[pd.DataFrame] = []
    for stratum, take in target.items():
        take = int(take)
        if take <= 0:
            continue
        sub = work[work["_strata"] == stratum]
        if sub.empty:
            continue
        sample = sub.sample(
            n=min(take, len(sub)), random_state=int(rng.integers(0, 10**9)),
        )
        pieces.append(sample)

    if not pieces:
        return work.sample(n=n, random_state=seed).drop(columns="_strata").reset_index(drop=True)

    out = pd.concat(pieces, ignore_index=True).drop(columns="_strata")
    if len(out) > n:
        out = out.sample(n=n, random_state=seed).reset_index(drop=True)
    elif len(out) < n:
        deficit = n - len(out)
        remaining = work.drop(columns="_strata").drop(out.index, errors="ignore")
        if len(remaining) >= deficit:
            extra = remaining.sample(n=deficit, random_state=seed)
            out = pd.concat([out, extra], ignore_index=True)
    return out.reset_index(drop=True)


def _allocate_across_splits(
    df: pd.DataFrame, total: int, seed: int,
) -> dict[str, int]:
    """Split `total` across train/val/test according to TRAIN/VAL/TEST_SAMPLE proportions."""
    if total <= 0 or df.empty:
        return {"train": 0, "val": 0, "test": 0}

    weights = {"train": TRAIN_SAMPLE, "val": VAL_SAMPLE, "test": TEST_SAMPLE}
    total_weight = sum(weights.values())
    raw = {k: total * w / total_weight for k, w in weights.items()}
    allocated = {k: int(np.floor(v)) for k, v in raw.items()}

    leftover = total - sum(allocated.values())
    fractions = sorted(
        ((k, raw[k] - allocated[k]) for k in raw),
        key=lambda kv: kv[1],
        reverse=True,
    )
    for k, _ in fractions:
        if leftover <= 0:
            break
        allocated[k] += 1
        leftover -= 1

    rng = np.random.default_rng(seed)
    for split_name, n in allocated.items():
        match = df["source_file"].str.contains(split_name, case=False, na=False)
        available = int(match.sum())
        if n > available:
            shortfall = n - available
            allocated[split_name] = available
            other_splits = [k for k in allocated if k != split_name]
            if other_splits:
                idx = int(rng.integers(0, len(other_splits)))
                allocated[other_splits[idx]] += shortfall
    return allocated


def _pick_within_category(
    df: pd.DataFrame, allocations: dict[str, int], seed: int,
) -> pd.DataFrame:
    """Stratify-sample the requested count per split from a category subset."""
    pieces: list[pd.DataFrame] = []
    for split_name, n in allocations.items():
        if n <= 0:
            continue
        match = df["source_file"].str.contains(split_name, case=False, na=False)
        sub = df[match]
        if sub.empty:
            continue
        pieces.append(_proportional_stratified_pick(sub, n, seed))
    if not pieces:
        return df.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=True)


def two_stage_stratified_sample(
    combined_df: pd.DataFrame,
    sample_size: int = SAMPLE_SIZE,
    named_categories: tuple[str, ...] = NAMED_CATEGORIES,
    min_per_named: int = MIN_PER_NAMED_CATEGORY,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Two-stage stratified sample.

    Stage 1: For each named category, take `min_per_named` rows stratified by
             (questionType, is_answerable) and allocated across train/val/test.
    Stage 2: Fill the remainder from the rest of the dataset, stratified by
             (questionType, is_answerable) and allocated across splits.

    Returns a DataFrame with stable `record_id` REC_001..REC_{sample_size} and
    a `q_bucket` column.
    """
    if combined_df.empty:
        raise ValueError("combined_df is empty")

    rng = np.random.default_rng(seed)

    # ---------- Stage 1 ----------
    stage1_pieces: list[pd.DataFrame] = []
    for category in named_categories:
        cat_df = combined_df[combined_df["category"] == category]
        if cat_df.empty:
            logger.warning("Named category %r missing from input data", category)
            continue
        if len(cat_df) < min_per_named:
            logger.warning(
                "Category %r has only %d rows (< floor %d); taking all of them",
                category, len(cat_df), min_per_named,
            )
        target = min(min_per_named, len(cat_df))
        allocations = _allocate_across_splits(
            cat_df, target, seed=int(rng.integers(0, 10**9)),
        )
        picked = _pick_within_category(
            cat_df, allocations, seed=int(rng.integers(0, 10**9)),
        )
        stage1_pieces.append(picked)

    stage1 = (
        pd.concat(stage1_pieces, ignore_index=True)
        if stage1_pieces else combined_df.iloc[0:0].copy()
    )

    # ---------- Stage 2 ----------
    remaining = combined_df.drop(index=stage1.index, errors="ignore")
    qid_used = set(stage1["qid"].tolist()) if "qid" in stage1.columns else set()
    if qid_used:
        remaining = remaining[~remaining["qid"].isin(qid_used)]

    stage2_target = sample_size - len(stage1)
    if stage2_target < 0:
        stage1 = stage1.head(sample_size)
        stage2_target = 0

    stage2 = combined_df.iloc[0:0].copy()
    if stage2_target > 0:
        allocations = _allocate_across_splits(
            remaining, stage2_target, seed=int(rng.integers(0, 10**9)),
        )
        for split_name, n in allocations.items():
            if n <= 0:
                continue
            match = remaining["source_file"].str.contains(split_name, case=False, na=False)
            sub = remaining[match]
            if sub.empty:
                continue
            stage2 = pd.concat(
                [stage2, _proportional_stratified_pick(sub, n, seed=int(rng.integers(0, 10**9)))],
                ignore_index=True,
            )

    sampled = pd.concat([stage1, stage2], ignore_index=True).reset_index(drop=True)
    if len(sampled) > sample_size:
        sampled = sampled.head(sample_size)

    sampled["record_id"] = [f"REC_{i + 1:03d}" for i in range(len(sampled))]
    sampled["q_bucket"] = sampled["questionText"].apply(assign_q_bucket)
    return sampled


# Backwards-compatible alias for callers expecting the v3/v4 entry point.
def stratified_sample_by_source(
    combined_df: pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """v5 default: two-stage stratification with named-category floor."""
    return two_stage_stratified_sample(combined_df, seed=seed)
