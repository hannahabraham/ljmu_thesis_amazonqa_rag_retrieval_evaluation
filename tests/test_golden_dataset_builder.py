"""Unit tests for golden dataset builder utilities."""

from __future__ import annotations

import pandas as pd
import pytest

from src.golden_dataset_builder import (
    JudgeResponse,
    build_draft_row,
    is_grounded,
    jeffreys_score,
    needs_judge_flag,
    parse_judge_response,
    score_candidates,
    validate_golden_consistency,
)


def test_jeffreys_smoothing() -> None:
    """Test Jeffreys smoothing for empty and strong positive votes."""
    assert jeffreys_score(0, 0) == 0.5
    assert jeffreys_score(10, 10) > 0.9


def test_grounding_jaccard_threshold() -> None:
    """Test answer grounding passes when evidence has sufficient overlap."""
    grounded, _, score = is_grounded(
        "Yes the battery lasts eight hours",
        ["The battery lasts about 8 hours according to my measurements."],
    )

    assert grounded
    assert score > 0.1


def test_score_candidates_orders_by_jeffreys() -> None:
    """Test candidate answers are sorted by Jeffreys score."""
    answers = [
        {"answerText": "no", "helpful": 1, "unhelpful": 0},
        {"answerText": "yes", "helpful": 20, "unhelpful": 1},
    ]

    scored = score_candidates(answers)

    assert scored[0]["text"] == "yes"


def test_needs_judge_when_top_ungrounded() -> None:
    """Test judge is needed when top candidate is ungrounded."""
    scored = [
        {"jeffreys": 0.9, "total": 5},
        {"jeffreys": 0.6, "total": 4},
    ]

    assert needs_judge_flag(scored, top_grounded=False, is_answerable=1) is True


def test_no_judge_when_top_clean() -> None:
    """Test judge is not needed when top candidate is grounded and clear."""
    scored = [
        {"jeffreys": 0.9, "total": 5},
        {"jeffreys": 0.5, "total": 3},
    ]

    assert needs_judge_flag(scored, top_grounded=True, is_answerable=1) is False


def test_build_draft_row_grounded(
    sample_record: dict,
    fake_kb: pd.DataFrame,
) -> None:
    """Test draft row selects grounded evidence."""
    row = pd.Series(sample_record)

    draft = build_draft_row(row, fake_kb)

    assert draft["evidence_doc_id"] in {"KB_00001", "KB_00002"}
    assert draft["selection_method"] in {"jeffreys", "grounded_pick"}


def test_parse_judge_response_strips_fences() -> None:
    """Test judge response parsing removes Markdown code fences."""
    raw_response = (
        "```json\n"
        '{"golden_answer": "yes", "evidence_doc_id": "KB_00001", '
        '"judge_confidence": 0.9, "reasoning": "ok"}\n'
        "```"
    )

    parsed = parse_judge_response(raw_response)

    assert isinstance(parsed, JudgeResponse)
    assert parsed.judge_confidence == 0.9


def test_validate_golden_passes(fake_kb: pd.DataFrame) -> None:
    """Test valid golden rows pass consistency validation and are marked ok."""
    golden = pd.DataFrame(
        [
            {
                "golden_id": "G_001",
                "golden_answer": "Yes, waterproof",
                "answerability": 1,
                "evidence_doc_id": "KB_00001",
                "evidence_text": "Product 1 is fully waterproof to 10m.",
            },
        ]
    )

    annotated = validate_golden_consistency(golden, fake_kb)

    assert list(annotated["validation_status"]) == ["ok"]


def test_validate_golden_fails_missing_doc(fake_kb: pd.DataFrame) -> None:
    """Test validation fails when evidence document is missing from KB."""
    golden = pd.DataFrame(
        [
            {
                "golden_id": "G_001",
                "golden_answer": "Yes",
                "answerability": 1,
                "evidence_doc_id": "KB_99999",
                "evidence_text": "anything",
            },
        ]
    )

    with pytest.raises(ValueError, match="not in KB"):
        validate_golden_consistency(golden, fake_kb)


def test_validate_golden_fails_drift(fake_kb: pd.DataFrame) -> None:
    """Test validation fails when evidence text has drifted."""
    golden = pd.DataFrame(
        [
            {
                "golden_id": "G_001",
                "golden_answer": "Yes",
                "answerability": 1,
                "evidence_doc_id": "KB_00001",
                "evidence_text": "DIFFERENT TEXT",
            },
        ]
    )

    with pytest.raises(ValueError, match="drifted"):
        validate_golden_consistency(golden, fake_kb)


def test_validate_golden_flags_unanswerable_with_evidence(
    fake_kb: pd.DataFrame,
) -> None:
    """Unanswerable rows carrying evidence are flagged, not raised."""
    golden = pd.DataFrame(
        [
            {
                "golden_id": "G_001",
                "golden_answer": "[UNANSWERABLE]",
                "answerability": 0,
                "evidence_doc_id": "KB_00001",
                "evidence_text": "irrelevant",
            },
        ]
    )

    annotated = validate_golden_consistency(golden, fake_kb)

    assert list(annotated["validation_status"]) == ["unanswerable_with_evidence"]


def test_validate_golden_flags_answerability_label_mismatch(
    fake_kb: pd.DataFrame,
) -> None:
    """Answerability/label disagreements are flagged as soft warnings."""
    golden = pd.DataFrame(
        [
            {
                "golden_id": "G_001",
                "golden_answer": "[UNANSWERABLE]",
                "answerability": 1,
                "evidence_doc_id": None,
                "evidence_text": None,
            },
        ]
    )

    annotated = validate_golden_consistency(golden, fake_kb)

    assert list(annotated["validation_status"]) == ["label_mismatch"]