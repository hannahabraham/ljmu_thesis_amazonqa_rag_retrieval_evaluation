"""Standardise raw AmazonQA splits to a common schema."""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

from src.utils.io import parse_list_field

LOGGER = logging.getLogger(__name__)

LIST_FIELDS = (
    "review_snippets",
    "top_sentences_IR",
    "top_review_helpful",
    "top_review_wilson",
    "answers",
)

_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _strip_surrogates(value: Any) -> Any:
    """Recursively remove lone Unicode surrogates."""
    if isinstance(value, str):
        return _SURROGATE_RE.sub("", value)

    if isinstance(value, dict):
        return {
            key: _strip_surrogates(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _strip_surrogates(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            _strip_surrogates(item)
            for item in value
        )

    return value


def _normalise_answerability(value: object) -> int | None:
    """Map common answerability values to 0 or 1."""
    if value is None:
        return None

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        return int(bool(value))

    text = str(value).strip().lower()

    if text in {"1", "true", "yes", "y", "answerable"}:
        return 1

    if text in {"0", "false", "no", "n", "unanswerable"}:
        return 0

    return None


def standardize_split(
    dataframe: pd.DataFrame,
    source_file: str,
) -> pd.DataFrame:
    """Standardise one raw AmazonQA split."""
    output = dataframe.copy()
    output["source_file"] = source_file

    for field_name in LIST_FIELDS:
        if field_name in output.columns:
            output[field_name] = output[field_name].apply(
                parse_list_field
            )
        else:
            output[field_name] = [
                []
                for _ in range(len(output))
            ]

    if "is_answerable" in output.columns:
        output["is_answerable"] = output["is_answerable"].apply(
            _normalise_answerability
        )
    else:
        output["is_answerable"] = None

    if "questionType" in output.columns:
        output["questionType"] = (
            output["questionType"]
            .astype(str)
            .str.lower()
            .str.strip()
        )
    else:
        output["questionType"] = "unknown"

    output["n_answers"] = output["answers"].apply(len)
    output["n_snippets"] = output["review_snippets"].apply(len)

    if "category" in output.columns:
        output["category"] = (
            output["category"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
        )
    else:
        output["category"] = "unknown"

    if "questionText" in output.columns:
        output["questionText"] = output["questionText"].astype(str)

    for column_name in output.columns:
        if output[column_name].dtype == "object":
            output[column_name] = output[column_name].map(
                _strip_surrogates
            )

    return output