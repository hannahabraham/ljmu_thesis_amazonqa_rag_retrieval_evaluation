"""Two-stage stratified sampling for the evaluation dataset.

Stage 1 guarantees minimum coverage for the named categories defined in
``NAMED_CATEGORIES`` using stratified sampling by question type and
answerability.

Stage 2 fills the remaining sample budget from the wider dataset while
preserving train/validation/test proportions and stratum balance.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from config.settings import (
    MIN_PER_NAMED_CATEGORY,
    NAMED_CATEGORIES,
    QUESTION_LENGTH_BUCKETS,
    RANDOM_SEED,
    SAMPLE_QUOTAS,
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
    # Filter by qid, not by positional index. stage1 has a fresh RangeIndex
    # after concat(ignore_index=True), so dropping by stage1.index against
    # combined_df would remove rows by coincident integer position rather
    # than the actual stage-1 selections.
    used_qids: set[str] = (
        set(stage1["qid"].tolist())
        if "qid" in stage1.columns
        else set()
    )

    if used_qids:
        remaining = combined_df[
            ~combined_df["qid"].isin(used_qids)
        ].copy()
    else:
        remaining = combined_df.copy()

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


# ----------------------------------------------------------------------
# Quota-driven sampler (used by step05).
# ----------------------------------------------------------------------

_FALLBACK_SPLIT_ORDER: dict[str, tuple[str, ...]] = {
    "train": ("val", "test"),
    "val": ("train", "test"),
    "test": ("val", "train"),
}


def _matches_split(series: pd.Series, split_name: str) -> pd.Series:
    """Return a boolean mask selecting rows belonging to a split."""
    return series.str.contains(split_name, case=False, na=False)


def _draw_rows(
    pool: pd.DataFrame,
    needed: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample up to ``needed`` rows from ``pool`` without replacement."""
    if needed <= 0 or pool.empty:
        return pool.iloc[0:0].copy()

    take = min(needed, len(pool))
    return pool.sample(
        n=take,
        random_state=int(rng.integers(0, 10**9)),
    )


def _do_one_swap(
    sampled: pd.DataFrame,
    selected_keys: set[tuple[str, str]],
    swap_out_candidates: pd.DataFrame,
    combined_df: pd.DataFrame,
    swap_in_predicate,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, bool]:
    """Perform one within-cell swap. Returns (sampled, swapped_flag).

    ``swap_in_predicate(swap_out_row) -> pd.DataFrame`` returns the pool of
    population rows eligible to swap into the cell vacated by
    ``swap_out_row``.
    """
    shuffled = swap_out_candidates.sample(
        frac=1.0,
        random_state=int(rng.integers(0, 10**9)),
    )

    for swap_out_idx, swap_out in shuffled.iterrows():
        cell_pool = swap_in_predicate(swap_out)
        if cell_pool.empty:
            continue

        pool_keys = pd.Series(
            list(
                zip(
                    cell_pool["qid"].astype(str),
                    cell_pool["source_file"].astype(str),
                )
            ),
            index=cell_pool.index,
        )
        cell_pool = cell_pool[~pool_keys.isin(selected_keys)]
        if cell_pool.empty:
            continue

        swap_in = cell_pool.sample(
            n=1,
            random_state=int(rng.integers(0, 10**9)),
        ).iloc[0]

        sampled = sampled.drop(swap_out_idx).reset_index(drop=True)
        selected_keys.discard(
            (str(swap_out["qid"]), str(swap_out["source_file"]))
        )
        sampled = pd.concat(
            [sampled, swap_in.to_frame().T],
            ignore_index=True,
        )
        selected_keys.add(
            (str(swap_in["qid"]), str(swap_in["source_file"]))
        )
        return sampled, True

    return sampled, False


def _rebalance_named_categories(
    sampled: pd.DataFrame,
    combined_df: pd.DataFrame,
    named_categories: Sequence[str],
    target: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Rebalance the sample so every named category has exactly ``target`` rows.

    Swaps stay inside the same ``(questionType, is_answerable, source_file)``
    cell, so cell-level quotas are unchanged — only which category appears
    within each cell changes.

    Pass 1 swaps over-represented named categories down to ``target`` (e.g.
    Electronics 52→30, donating slots to 'other').

    Pass 2 swaps under-represented named categories up from 'other' to
    ``target`` (e.g. Health 8→30, consuming 'other' slots).
    """
    sampled = sampled.reset_index(drop=True).copy()
    named_set = set(named_categories)

    selected_keys: set[tuple[str, str]] = set(
        zip(sampled["qid"].astype(str), sampled["source_file"].astype(str))
    )

    def _same_cell(swap_out: pd.Series) -> pd.Series:
        return (
            (combined_df["questionType"].astype(str) == str(swap_out["questionType"]))
            & (combined_df["is_answerable"] == swap_out["is_answerable"])
            & (combined_df["source_file"] == swap_out["source_file"])
        )

    # --- Pass 1: swap over-represented named categories DOWN ---
    for category in named_categories:
        while True:
            current = int((sampled["category"] == category).sum())
            excess = current - target
            if excess <= 0:
                break

            swap_out_pool = sampled[sampled["category"] == category]

            def predicate(swap_out: pd.Series) -> pd.DataFrame:
                return combined_df[
                    _same_cell(swap_out)
                    & (~combined_df["category"].isin(named_set))
                ]

            sampled, swapped = _do_one_swap(
                sampled,
                selected_keys,
                swap_out_pool,
                combined_df,
                predicate,
                rng,
            )
            if not swapped:
                LOGGER.warning(
                    "Named-category %-30s n=%d (target=%d, %d above — "
                    "no compatible swap-down available)",
                    category, current, target, excess,
                )
                break

    # --- Pass 2: swap under-represented named categories UP from 'other' ---
    for category in named_categories:
        while True:
            current = int((sampled["category"] == category).sum())
            deficit = target - current
            if deficit <= 0:
                LOGGER.info(
                    "Named-category %-30s n=%d (target=%d, OK)",
                    category, current, target,
                )
                break

            swap_out_pool = sampled[~sampled["category"].isin(named_set)]
            if swap_out_pool.empty:
                LOGGER.warning(
                    "Cannot top up %s: no 'other' rows left to swap out "
                    "(n=%d, %d below target)",
                    category, current, deficit,
                )
                break

            def predicate(swap_out: pd.Series, _cat: str = category) -> pd.DataFrame:
                return combined_df[
                    _same_cell(swap_out)
                    & (combined_df["category"] == _cat)
                ]

            sampled, swapped = _do_one_swap(
                sampled,
                selected_keys,
                swap_out_pool,
                combined_df,
                predicate,
                rng,
            )
            if not swapped:
                LOGGER.warning(
                    "Named-category %-30s n=%d (target=%d, %d below — "
                    "no compatible swap-up available)",
                    category, current, target, deficit,
                )
                break

    return sampled.reset_index(drop=True)


def quota_stratified_sample(
    combined_df: pd.DataFrame,
    quotas: dict[tuple[str, int], dict[str, int]] | None = None,
    seed: int = RANDOM_SEED,
    named_category_floor: int = MIN_PER_NAMED_CATEGORY,
    named_categories: Sequence[str] = NAMED_CATEGORIES,
) -> pd.DataFrame:
    """Sample rows to match an explicit ``(questionType, is_answerable, split)`` quota.

    The proportional sampler under-represents yes/no questions because they
    are only ~15% of the AmazonQA population. This sampler instead draws a
    pre-declared number of rows per cell, so the final 200-row evaluation
    set has enough yes/no and unanswerable examples to power those
    sub-analyses.

    If a cell underfills (population too small), the deficit is redirected
    to other splits within the **same questionType + answerability**, so
    yes/no never gets backfilled with descriptive rows and answerable
    never gets backfilled with unanswerable rows.

    After the cell-level draw, ``_top_up_named_categories`` swaps 'other'
    rows for named-category rows within the same cell so each named
    category meets ``named_category_floor``. The cell-level quota is
    preserved exactly; only which category appears within the cell changes.
    """
    if combined_df.empty:
        raise ValueError("combined_df is empty")

    if "qid" not in combined_df.columns:
        raise ValueError("combined_df must contain a 'qid' column")

    quotas = SAMPLE_QUOTAS if quotas is None else quotas

    rng = np.random.default_rng(seed)
    selected_pieces: list[pd.DataFrame] = []
    # (question_type, answerability, split, original_target, adjusted_target, actual_drawn)
    fill_report: list[tuple[str, int, str, int, int, int]] = []

    for (question_type, answerability), split_targets in quotas.items():
        type_mask = combined_df["questionType"].astype(str) == question_type
        ans_mask = combined_df["is_answerable"] == answerability
        cell_df = combined_df[type_mask & ans_mask]

        per_split_pools: dict[str, pd.DataFrame] = {
            split_name: cell_df[_matches_split(cell_df["source_file"], split_name)]
            for split_name in split_targets
        }

        carry_over: dict[str, int] = dict.fromkeys(split_targets, 0)

        for split_name, target in split_targets.items():
            pool = per_split_pools[split_name]
            available = len(pool)

            if available >= target:
                continue

            shortfall = target - available
            for fallback_split in _FALLBACK_SPLIT_ORDER.get(
                split_name, ()
            ):
                if fallback_split not in split_targets:
                    continue
                carry_over[fallback_split] += shortfall
                LOGGER.warning(
                    "Cell (%s, ans=%d, split=%s) short by %d rows; "
                    "redirecting demand to split=%s",
                    question_type,
                    answerability,
                    split_name,
                    shortfall,
                    fallback_split,
                )
                break
            else:
                LOGGER.warning(
                    "Cell (%s, ans=%d, split=%s) short by %d rows and "
                    "no fallback split available; final sample will "
                    "underfill this cell.",
                    question_type,
                    answerability,
                    split_name,
                    shortfall,
                )

        for split_name, target in split_targets.items():
            pool = per_split_pools[split_name]
            adjusted_target = min(
                target + carry_over[split_name],
                len(pool),
            )

            drawn = _draw_rows(pool, adjusted_target, rng)
            selected_pieces.append(drawn)

            fill_report.append(
                (
                    question_type,
                    answerability,
                    split_name,
                    target,
                    adjusted_target,
                    len(drawn),
                )
            )

    if not selected_pieces or all(piece.empty for piece in selected_pieces):
        raise RuntimeError("quota_stratified_sample drew zero rows")

    # Look back by the composite (qid, source_file) key — qid alone is not
    # unique across train/val/test in AmazonQA, so isin-on-qid would silently
    # rebind val/test selections to their train twin.
    sampled = pd.concat(selected_pieces, ignore_index=True)
    sampled = sampled.drop_duplicates(subset=["qid", "source_file"])

    if named_category_floor > 0 and named_categories:
        LOGGER.info(
            "Rebalancing named categories to target=%d each:",
            named_category_floor,
        )
        sampled = _rebalance_named_categories(
            sampled,
            combined_df,
            named_categories,
            named_category_floor,
            rng,
        )

    sort_key = pd.Categorical(
        sampled["source_file"].astype(str).str.lower(),
        categories=["train", "val", "test"],
        ordered=True,
    )
    sampled = (
        sampled.assign(_split_order=sort_key)
        .sort_values(["_split_order", "questionType", "is_answerable", "qid"])
        .drop(columns="_split_order")
        .reset_index(drop=True)
    )

    sampled["record_id"] = [
        f"REC_{idx + 1:03d}" for idx in range(len(sampled))
    ]
    sampled["q_bucket"] = sampled["questionText"].apply(assign_q_bucket)

    for (
        question_type,
        ans_flag,
        split_name,
        target,
        adjusted_target,
        actual,
    ) in fill_report:
        if actual == adjusted_target == target:
            marker = "OK"
        elif adjusted_target > target and actual == adjusted_target:
            marker = "OVERFILL (absorbed redirect)"
        elif actual < adjusted_target:
            marker = "UNDERFILL"
        else:
            marker = "UNDER vs original (donated to sibling)"
        LOGGER.info(
            "  quota %-12s ans=%d split=%-5s target=%2d adjusted=%2d actual=%2d  [%s]",
            question_type,
            ans_flag,
            split_name,
            target,
            adjusted_target,
            actual,
            marker,
        )

    LOGGER.info(
        "Quota sampler produced %d rows (target %d)",
        len(sampled),
        sum(s for cell in quotas.values() for s in cell.values()),
    )

    return sampled
