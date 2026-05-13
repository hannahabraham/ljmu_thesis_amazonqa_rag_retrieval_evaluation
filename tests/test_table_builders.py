"""Schema-completeness tests for Results Sheet table builders."""

from __future__ import annotations

import pandas as pd

from config.settings import NAMED_CATEGORIES, PIPELINE_KEYS
from src.evaluation.table_builders import (
    TABLE1_COLUMNS,
    TABLE2_COLUMNS,
    TABLE3_COLUMNS,
    TABLE4_COLUMNS,
    TABLE6_COLUMNS,
    TABLE7_COLUMNS,
    build_pairwise_wilcoxon,
    build_table1_overall,
    build_table2_depth,
    build_table3_category,
    build_table4_length,
    build_table6_answerability,
    build_table7_final_ranking,
)


def _make_per_q(seed: int = 0) -> pd.DataFrame:
    """Create a synthetic per-question evaluation DataFrame."""
    rng = __import__("numpy").random.default_rng(seed)

    rows = []
    record_counter = 0

    for pipeline in PIPELINE_KEYS:
        for k_value in (1, 3, 5, 10):
            for category in (*NAMED_CATEGORIES, "Other"):
                for bucket in ("short", "medium", "long"):
                    # Four records per combination to keep totals manageable.
                    for index in range(4):
                        record_counter += 1

                        is_answerable = bool(index % 2 == 0)
                        refused = (
                            not is_answerable
                            if index % 4 == 0
                            else False
                        )
                        token_f1 = float(rng.uniform(0.0, 1.0))

                        rows.append(
                            {
                                "pipeline": pipeline,
                                "k": k_value,
                                "golden_id": f"G_{record_counter:05d}",
                                "record_id": f"REC_{record_counter:05d}",
                                "asin": f"B{record_counter:06d}",
                                "category": category,
                                "q_bucket": bucket,
                                "question_type": "yesno",
                                "is_answerable": is_answerable,
                                "evidence_doc_id": (
                                    "KB_00001"
                                    if is_answerable
                                    else None
                                ),
                                "retrieved_doc_ids": ["KB_00001"],
                                "retrieved_context": ["ctx"],
                                "question": "Q?",
                                "gold_answer": (
                                    "A"
                                    if is_answerable
                                    else "[UNANSWERABLE]"
                                ),
                                "generated_answer": "A",
                                "refused": refused,
                                "em": int(token_f1 > 0.8),
                                "token_f1": token_f1,
                                "is_correct": (
                                    (
                                        is_answerable
                                        and not refused
                                        and token_f1 >= 0.5
                                    )
                                    or (
                                        not is_answerable
                                        and refused
                                    )
                                ),
                                "retrieval_ms": 10.0,
                                "generation_ms": 100.0,
                                "total_ms": 110.0,
                                "faithfulness": float(
                                    rng.uniform(0.4, 1.0)
                                ),
                                "context_precision": float(
                                    rng.uniform(0.4, 1.0)
                                ),
                                "context_recall": float(
                                    rng.uniform(0.4, 1.0)
                                ),
                            }
                        )

    return pd.DataFrame(rows)


def _make_retrieval() -> pd.DataFrame:
    """Create a synthetic retrieval metrics DataFrame."""
    rows = []

    for pipeline in PIPELINE_KEYS:
        for k_value in (1, 3, 5, 10):
            rows.append(
                {
                    "pipeline": pipeline,
                    "k": k_value,
                    "recall_at_k": 0.5,
                    "mrr": 0.4,
                }
            )

    return pd.DataFrame(rows)


def _make_ragas() -> pd.DataFrame:
    """Create a synthetic RAGAS metrics DataFrame."""
    rows = []

    for pipeline in PIPELINE_KEYS:
        for k_value in (1, 3, 5, 10):
            rows.append(
                {
                    "pipeline": pipeline,
                    "k": k_value,
                    "faithfulness": 0.7,
                    "context_precision": 0.7,
                    "context_recall": 0.7,
                }
            )

    return pd.DataFrame(rows)


def test_table1_schema_and_row_count() -> None:
    """Test Table 1 schema and expected row count."""
    per_question = _make_per_q()

    table = build_table1_overall(
        per_question,
        ragas_df=_make_ragas(),
    )

    assert list(table.columns) == TABLE1_COLUMNS
    assert len(table) == len(PIPELINE_KEYS)


def test_table2_schema_and_row_count() -> None:
    """Test Table 2 schema and expected row count."""
    per_question = _make_per_q()

    table = build_table2_depth(
        per_question,
        retrieval_df=_make_retrieval(),
        ragas_df=_make_ragas(),
    )

    assert list(table.columns) == TABLE2_COLUMNS
    assert len(table) == len(PIPELINE_KEYS) * 4


def test_table3_schema_and_row_count() -> None:
    """Test Table 3 schema and expected row count."""
    per_question = _make_per_q()

    table = build_table3_category(
        per_question,
        ragas_df=_make_ragas(),
    )

    assert list(table.columns) == TABLE3_COLUMNS
    assert len(table) == len(NAMED_CATEGORIES) * len(PIPELINE_KEYS)


def test_table3_excludes_non_named_categories() -> None:
    """Test Table 3 excludes categories outside the named set."""
    per_question = _make_per_q()

    table = build_table3_category(
        per_question,
        ragas_df=_make_ragas(),
    )

    assert set(table["Product Category"].unique()) == set(
        NAMED_CATEGORIES
    )


def test_table4_schema_and_row_count() -> None:
    """Test Table 4 schema and expected row count."""
    per_question = _make_per_q()

    table = build_table4_length(
        per_question,
        ragas_df=_make_ragas(),
    )

    assert list(table.columns) == TABLE4_COLUMNS
    assert len(table) == 3 * len(PIPELINE_KEYS)


def test_table6_schema_and_row_count() -> None:
    """Test Table 6 schema and expected row count."""
    per_question = _make_per_q()

    table = build_table6_answerability(
        per_question,
        ragas_df=_make_ragas(),
    )

    assert list(table.columns) == TABLE6_COLUMNS
    assert len(table) == len(PIPELINE_KEYS)


def test_table7_schema_and_row_count() -> None:
    """Test Table 7 schema, row count, and ranking uniqueness."""
    per_question = _make_per_q()

    table = build_table7_final_ranking(
        per_question,
        ragas_df=_make_ragas(),
    )

    assert list(table.columns) == TABLE7_COLUMNS
    assert len(table) == len(PIPELINE_KEYS)

    # Ranks should be consecutive with no duplicates.
    assert sorted(table["Rank"].tolist()) == list(
        range(1, len(PIPELINE_KEYS) + 1)
    )


def test_pairwise_wilcoxon_emits_c5_2_pairs_per_metric() -> None:
    """Test pairwise Wilcoxon emits all expected pipeline pairs."""
    per_question = _make_per_q()

    pairwise = build_pairwise_wilcoxon(
        per_question,
        metrics=("token_f1", "faithfulness"),
    )

    expected_pairs = (
        len(PIPELINE_KEYS) * (len(PIPELINE_KEYS) - 1)
    ) // 2

    assert len(pairwise) == 2 * expected_pairs