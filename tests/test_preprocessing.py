"""Unit tests for dataset preprocessing utilities."""

from __future__ import annotations

import pandas as pd

from src.preprocessing import standardize_split


def test_standardize_parses_list_fields_and_normalises_answerability() -> None:
    """Test parsing of list fields and answerability normalisation."""
    raw_dataframe = pd.DataFrame(
        [
            {
                "qid": "Q1",
                "asin": "A1",
                "questionText": "Is it waterproof?",
                "questionType": "YesNo",
                "is_answerable": "yes",
                "answers": (
                    '[{"answerText": "yes", '
                    '"helpful": 5, '
                    '"unhelpful": 0}]'
                ),
                "review_snippets": '["a snippet"]',
                "category": "Electronics",
            },
        ]
    )

    output = standardize_split(raw_dataframe, source_file="train")

    assert output.loc[0, "questionType"] == "yesno"
    assert output.loc[0, "is_answerable"] == 1
    assert output.loc[0, "n_answers"] == 1
    assert output.loc[0, "n_snippets"] == 1
    assert output.loc[0, "source_file"] == "train"


def test_standardize_handles_missing_columns() -> None:
    """Test preprocessing handles missing optional columns gracefully."""
    raw_dataframe = pd.DataFrame(
        [
            {
                "qid": "Q1",
                "asin": "A1",
                "questionText": "X",
            },
        ]
    )

    output = standardize_split(raw_dataframe, source_file="val")

    assert output.loc[0, "category"] == "unknown"
    assert output.loc[0, "questionType"] == "unknown"
    assert output.loc[0, "n_answers"] == 0
    assert output.loc[0, "n_snippets"] == 0