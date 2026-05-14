"""Two-stage stratified sampling for the evaluation dataset.

Stage 1 guarantees minimum coverage for the named categories defined in
``NAMED_CATEGORIES`` using stratified sampling by question type and
answerability.

Stage 2 fills the remaining sample budget from the wider dataset while
preserving train/validation/test proportions and stratum balance.
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

LOGGER = logging.getLogger(__name__)


def assign_q_bucket(question: str) -> str:
    """Assign a question-length bucket based on token count."""
    short_max, medium_max = QUESTION_LENGTH_BUCKETS

    token_count = len(str(question).split())

    if token_count <= short_max:
        return "short"

    if token_count <= medium_max:
        return "medium"

    return "long"


def _proportional_stratified_pick(
    dataframe: pd.DataFrame,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    """Sample rows proportionally by question type and answerability."""
    if dataframe.empty or sample_size <= 0:
        return dataframe.iloc[0:0].copy()

    if len(dataframe) <= sample_size:
        return dataframe.sample(
            frac=1.0,
            random_state=seed,
        ).reset_index(drop=True)

    working_df = dataframe.copy()

    if "questionType" not in working_df.columns:
        working_df["questionType"] = "unknown"

    if "is_answerable" not in working_df.columns:
        working_df["is_answerable"] = -1

    working_df["_strata"] = (
        working_df["questionType"].astype(str)
        + "|"
        + working_df["is_answerable"].astype(str)
    )

    proportions = working_df["_strata"].value_counts(
        normalize=True
    )

    rng = np.random.default_rng(seed)

    target_counts = (
        proportions * sample_size
    ).round().astype(int)

    difference = sample_size - int(target_counts.sum())

    if difference != 0:
        for stratum in proportions.index:
            if difference == 0:
                break

            step = 1 if difference > 0 else -1
            new_value = int(target_counts[stratum]) + step

            if new_value >= 0:
                target_counts[stratum] = new_value
                difference -= step

    sampled_pieces: list[pd.DataFrame] = []

    for stratum, take_count in target_counts.items():
        take_count = int(take_count)

        if take_count <= 0:
            continue

        subset = working_df[
            working_df["_strata"] == stratum
        ]

        if subset.empty:
            continue

        sampled_subset = subset.sample(
            n=min(take_count, len(subset)),
            random_state=int(rng.integers(0, 10**9)),
        )

        sampled_pieces.append(sampled_subset)

    if not sampled_pieces:
        return (
            working_df.sample(
                n=sample_size,
                random_state=seed,
            )
            .drop(columns="_strata")
            .reset_index(drop=True)
        )

    sampled = pd.concat(
        sampled_pieces,
        ignore_index=True,
    ).drop(columns="_strata")

    if len(sampled) > sample_size:
        sampled = sampled.sample(
            n=sample_size,
            random_state=seed,
        ).reset_index(drop=True)

    elif len(sampled) < sample_size:
        deficit = sample_size - len(sampled)

        remaining = (
            working_df.drop(columns="_strata")
            .drop(sampled.index, errors="ignore")
        )

        if len(remaining) >= deficit:
            extra = remaining.sample(
                n=deficit,
                random_state=seed,
            )

            sampled = pd.concat(
                [sampled, extra],
                ignore_index=True,
            )

    return sampled.reset_index(drop=True)


def _allocate_across_splits(
    dataframe: pd.DataFrame,
    total: int,
    seed: int,
) -> dict[str, int]:
    """Allocate sample counts across train, validation, and test splits."""
    if total <= 0 or dataframe.empty:
        return {
            "train": 0,
            "val": 0,
            "test": 0,
        }

    weights = {
        "train": TRAIN_SAMPLE,
        "val": VAL_SAMPLE,
        "test": TEST_SAMPLE,
    }

    total_weight = sum(weights.values())

    raw_allocations = {
        split_name: total * weight / total_weight
        for split_name, weight in weights.items()
    }

    allocations = {
        split_name: int(np.floor(value))
        for split_name, value in raw_allocations.items()
    }

    leftover = total - sum(allocations.values())

    fractions = sorted(
        (
            (
                split_name,
                raw_allocations[split_name]
                - allocations[split_name],
            )
            for split_name in raw_allocations
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    for split_name, _ in fractions:
        if leftover <= 0:
            break

        allocations[split_name] += 1
        leftover -= 1

    rng = np.random.default_rng(seed)

    for split_name, allocation in allocations.items():
        match = dataframe["source_file"].str.contains(
            split_name,
            case=False,
            na=False,
        )

        available = int(match.sum())

        if allocation > available:
            shortfall = allocation - available

            allocations[split_name] = available

            alternative_splits = [
                name
                for name in allocations
                if name != split_name
            ]

            if alternative_splits:
                index = int(
                    rng.integers(0, len(alternative_splits))
                )

                allocations[
                    alternative_splits[index]
                ] += shortfall

    return allocations


def _pick_within_category(
    dataframe: pd.DataFrame,
    allocations: dict[str, int],
    seed: int,
) -> pd.DataFrame:
    """Sample the requested number of rows per dataset split."""
    sampled_pieces: list[pd.DataFrame] = []

    for split_name, sample_size in allocations.items():
        if sample_size <= 0:
            continue

        match = dataframe["source_file"].str.contains(
            split_name,
            case=False,
            na=False,
        )

        subset = dataframe[match]

        if subset.empty:
            continue

        sampled_pieces.append(
            _proportional_stratified_pick(
                subset,
                sample_size,
                seed,
            )
        )

    if not sampled_pieces:
        return dataframe.iloc[0:0].copy()

    return pd.concat(sampled_pieces, ignore_index=True)


def two_stage_stratified_sample(
    combined_df: pd.DataFrame,
    sample_size: int = SAMPLE_SIZE,
    named_categories: tuple[str, ...] = NAMED_CATEGORIES,
    min_per_named: int = MIN_PER_NAMED_CATEGORY,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Create a two-stage stratified evaluation sample."""
    if combined_df.empty:
        raise ValueError("combined_df is empty")

    rng = np.random.default_rng(seed)

    stage1_pieces: list[pd.DataFrame] = []

    # ------------------------------------------------------------------
    # Stage 1: Guarantee minimum representation for named categories.
    # ------------------------------------------------------------------
    for category in named_categories:
        category_df = combined_df[
            combined_df["category"] == category
        ]

        if category_df.empty:
            LOGGER.warning(
                "Named category %r missing from input data",
                category,
            )
            continue

        if len(category_df) < min_per_named:
            LOGGER.warning(
                (
                    "Category %r has only %d rows "
                    "(< floor %d); taking all available rows"
                ),
                category,
                len(category_df),
                min_per_named,
            )

        target_size = min(
            min_per_named,
            len(category_df),
        )

        allocations = _allocate_across_splits(
            category_df,
            target_size,
            seed=int(rng.integers(0, 10**9)),
        )

        sampled_category = _pick_within_category(
            category_df,
            allocations,
            seed=int(rng.integers(0, 10**9)),
        )

        stage1_pieces.append(sampled_category)

    stage1 = (
        pd.concat(stage1_pieces, ignore_index=True)
        if stage1_pieces
        else combined_df.iloc[0:0].copy()
    )

    # ------------------------------------------------------------------
    # Stage 2: Fill remaining slots from the wider dataset.
    # ------------------------------------------------------------------
    remaining = combined_df.drop(
        index=stage1.index,
        errors="ignore",
    )

    used_qids = (
        set(stage1["qid"].tolist())
        if "qid" in stage1.columns
        else set()
    )

    if used_qids:
        remaining = remaining[
            ~remaining["qid"].isin(used_qids)
        ]

    stage2_target = sample_size - len(stage1)

    if stage2_target < 0:
        stage1 = stage1.head(sample_size)
        stage2_target = 0

    stage2 = combined_df.iloc[0:0].copy()

    if stage2_target > 0:
        allocations = _allocate_across_splits(
            remaining,
            stage2_target,
            seed=int(rng.integers(0, 10**9)),
        )

        for split_name, split_size in allocations.items():
            if split_size <= 0:
                continue

            match = remaining["source_file"].str.contains(
                split_name,
                case=False,
                na=False,
            )

            subset = remaining[match]

            if subset.empty:
                continue

            stage2 = pd.concat(
                [
                    stage2,
                    _proportional_stratified_pick(
                        subset,
                        split_size,
                        seed=int(rng.integers(0, 10**9)),
                    ),
                ],
                ignore_index=True,
            )

    sampled = pd.concat(
        [stage1, stage2],
        ignore_index=True,
    ).reset_index(drop=True)

    if len(sampled) > sample_size:
        sampled = sampled.head(sample_size)

    sampled["record_id"] = [
        f"REC_{index + 1:03d}"
        for index in range(len(sampled))
    ]

    sampled["q_bucket"] = sampled["questionText"].apply(
        assign_q_bucket
    )

    return sampled


def stratified_sample_by_source(
    combined_df: pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Backward-compatible sampling entry point."""
    return two_stage_stratified_sample(
        combined_df,
        seed=seed,
    )