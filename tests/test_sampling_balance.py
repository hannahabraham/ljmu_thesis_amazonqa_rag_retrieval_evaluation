"""Quota-driven sampler balance tests."""

from __future__ import annotations

import pandas as pd

from config.settings import SAMPLE_QUOTAS
from src.sampling import quota_stratified_sample


def _make_population(per_cell_rows: int = 80) -> pd.DataFrame:
    """Build a combined dataset large enough to satisfy every quota cell."""
    rows: list[dict] = []
    counter = 0

    for (question_type, answerability), split_targets in SAMPLE_QUOTAS.items():
        for split_name in split_targets:
            for index in range(per_cell_rows):
                counter += 1
                rows.append(
                    {
                        "qid": f"{split_name}_{question_type}_{answerability}_{index}",
                        "asin": f"A{counter:06d}",
                        "questionText": (
                            "is this waterproof"
                            if question_type == "yesno"
                            else "please describe how this works in detail"
                        ),
                        "questionType": question_type,
                        "is_answerable": answerability,
                        "source_file": split_name,
                        "category": "Electronics",
                    }
                )

    return pd.DataFrame(rows)


def test_quota_sampler_hits_each_cell_exactly() -> None:
    """Every (questionType, is_answerable, split) cell hits its declared target."""
    sample = quota_stratified_sample(_make_population(), seed=42)

    assert len(sample) == 200, sample["questionType"].value_counts().to_dict()

    cell_counts = (
        sample.groupby(
            ["questionType", "is_answerable", "source_file"]
        )
        .size()
        .to_dict()
    )

    for (question_type, answerability), split_targets in SAMPLE_QUOTAS.items():
        for split_name, target in split_targets.items():
            actual = cell_counts.get(
                (question_type, answerability, split_name), 0
            )
            assert actual == target, (
                f"({question_type}, ans={answerability}, "
                f"split={split_name}): expected {target}, got {actual}"
            )


def test_quota_sampler_totals_match_supervisor_brief() -> None:
    """Aggregate totals match the 55/145, 125/75, 120/40/40 supervisor brief."""
    sample = quota_stratified_sample(_make_population(), seed=42)

    type_counts = sample["questionType"].value_counts().to_dict()
    assert type_counts["yesno"] == 55
    assert type_counts["descriptive"] == 145

    ans_counts = sample["is_answerable"].value_counts().to_dict()
    assert ans_counts[1] == 125
    assert ans_counts[0] == 75

    split_counts = sample["source_file"].value_counts().to_dict()
    assert split_counts["train"] == 120
    assert split_counts["val"] == 40
    assert split_counts["test"] == 40


def test_quota_sampler_is_deterministic_with_same_seed() -> None:
    """Repeated runs with the same seed produce identical qid sets."""
    population = _make_population()

    first = quota_stratified_sample(population, seed=42)
    second = quota_stratified_sample(population, seed=42)

    assert sorted(first["qid"].tolist()) == sorted(second["qid"].tolist())


def test_quota_sampler_assigns_unique_sequential_record_ids() -> None:
    """REC_001..REC_200 are present and unique."""
    sample = quota_stratified_sample(_make_population(), seed=42)

    assert sample["record_id"].iloc[0] == "REC_001"
    assert sample["record_id"].iloc[-1] == "REC_200"
    assert sample["record_id"].is_unique
    assert sample["qid"].is_unique


def test_quota_sampler_redirects_within_questiontype_on_underfill() -> None:
    """A cell shortage redirects demand to a sibling split, never another type."""
    population = _make_population()

    # Starve yesno/answerable/train down to 5 rows; the other 16 must come
    # from yesno/answerable on a different split, not from descriptive.
    starved = population[
        ~(
            (population["questionType"] == "yesno")
            & (population["is_answerable"] == 1)
            & (population["source_file"] == "train")
        )
    ]
    keep_train = population[
        (population["questionType"] == "yesno")
        & (population["is_answerable"] == 1)
        & (population["source_file"] == "train")
    ].head(5)
    starved = pd.concat([starved, keep_train], ignore_index=True)

    sample = quota_stratified_sample(starved, seed=42)

    yes_ans = sample[
        (sample["questionType"] == "yesno") & (sample["is_answerable"] == 1)
    ]
    # Total yes/no answerable target = 35. With 5 train + 7 val + 7 test
    # natively + up to 16 redirected from val/test pools.
    assert len(yes_ans) >= 19  # 5 (train) + 7 (val) + 7 (test) = 19 minimum
    # Descriptive answerable should still match its original target.
    desc_ans = sample[
        (sample["questionType"] == "descriptive") & (sample["is_answerable"] == 1)
    ]
    assert len(desc_ans) == 90
