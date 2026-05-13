"""Unit tests for sampling utilities."""

from __future__ import annotations

import pandas as pd

from config.settings import MIN_PER_NAMED_CATEGORY, NAMED_CATEGORIES
from src.sampling import (
    assign_q_bucket,
    stratified_sample_by_source,
    two_stage_stratified_sample,
)


def _fake_combined(seed_offset: int = 0) -> pd.DataFrame:
    """Build a realistic combined dataset for stratified sampling tests."""
    rows = []
    counter = 0

    for split, row_count in [("train", 800), ("val", 300), ("test", 300)]:
        for index in range(row_count):
            counter += 1
            category_index = (index + seed_offset) % (len(NAMED_CATEGORIES) + 3)

            if category_index < len(NAMED_CATEGORIES):
                category = NAMED_CATEGORIES[category_index]
            else:
                category = f"Other_{category_index}"

            rows.append(
                {
                    "qid": f"{split}_{counter}",
                    "asin": f"A{counter:06d}",
                    "questionText": (
                        "is it good"
                        if index % 2 == 0
                        else "tell me more about this please"
                    ),
                    "questionType": "yesno" if index % 2 == 0 else "open",
                    "is_answerable": int(index % 3 != 0),
                    "source_file": split,
                    "category": category,
                }
            )

    return pd.DataFrame(rows)


def test_assign_q_bucket() -> None:
    """Test question text is assigned to the expected length bucket."""
    assert assign_q_bucket("is it waterproof") == "short"
    assert (
        assign_q_bucket("does this work with the latest android version please")
        == "medium"
    )
    assert assign_q_bucket(" ".join(["word"] * 14)) == "long"


def test_stratified_sample_by_source_assigns_record_ids() -> None:
    """Test source-stratified sampling creates unique sequential record IDs."""
    sample = stratified_sample_by_source(_fake_combined(), seed=42)

    assert len(sample) == 200
    assert sample["record_id"].iloc[0] == "REC_001"
    assert sample["record_id"].iloc[-1] == "REC_200"
    assert "q_bucket" in sample.columns
    assert sample["record_id"].is_unique


def test_two_stage_floors_named_categories() -> None:
    """Test two-stage sampling enforces floors for named categories."""
    sample = two_stage_stratified_sample(_fake_combined(), seed=42)

    assert len(sample) == 200

    counts = sample["category"].value_counts()

    for category in NAMED_CATEGORIES:
        assert counts.get(category, 0) >= MIN_PER_NAMED_CATEGORY, (
            f"{category}: only {counts.get(category, 0)} records "
            f"(< floor {MIN_PER_NAMED_CATEGORY})"
        )


def test_two_stage_deterministic_with_same_seed() -> None:
    """Test two-stage sampling is deterministic for a fixed seed."""
    dataframe = _fake_combined()

    first_sample = two_stage_stratified_sample(dataframe, seed=42)
    second_sample = two_stage_stratified_sample(dataframe, seed=42)

    assert first_sample["qid"].tolist() == second_sample["qid"].tolist()


def test_two_stage_no_duplicate_record_ids() -> None:
    """Test two-stage sampling returns unique record and question IDs."""
    sample = two_stage_stratified_sample(_fake_combined(), seed=42)

    assert sample["record_id"].is_unique
    assert sample["qid"].is_unique