"""Standardize raw AmazonQA splits to a common schema."""
from __future__ import annotations

import logging
import re

import pandas as pd

from src.utils.io import parse_list_field

logger = logging.getLogger(__name__)

LIST_FIELDS = (
    "review_snippets", "top_sentences_IR",
    "top_review_helpful", "top_review_wilson",
    "answers",
)

_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _strip_surrogates(value: object) -> object:
    """Recursively remove lone unicode surrogates so UTF-8 writers don't choke."""
    if isinstance(value, str):
        return _SURROGATE_RE.sub("", value)
    if isinstance(value, dict):
        return {k: _strip_surrogates(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_strip_surrogates(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_strip_surrogates(v) for v in value)
    return value


def _normalise_answerability(value: object) -> int | None:
    """Map various truthy/falsey representations to 0/1; None for unknown."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        return int(bool(value))
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "answerable"):
        return 1
    if text in ("0", "false", "no", "n", "unanswerable"):
        return 0
    return None


def standardize_split(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """Parse list fields, normalise answerability, lowercase questionType, derive counts."""
    out = df.copy()
    out["source_file"] = source_file

    for field in LIST_FIELDS:
        if field in out.columns:
            out[field] = out[field].apply(parse_list_field)
        else:
            out[field] = [[] for _ in range(len(out))]

    if "is_answerable" in out.columns:
        out["is_answerable"] = out["is_answerable"].apply(_normalise_answerability)
    else:
        out["is_answerable"] = None

    if "questionType" in out.columns:
        out["questionType"] = out["questionType"].astype(str).str.lower().str.strip()
    else:
        out["questionType"] = "unknown"

    out["n_answers"] = out["answers"].apply(len)
    out["n_snippets"] = out["review_snippets"].apply(len)

    if "category" in out.columns:
        out["category"] = out["category"].fillna("unknown").astype(str)
    else:
        out["category"] = "unknown"

    if "questionText" in out.columns:
        out["questionText"] = out["questionText"].astype(str)

    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].map(_strip_surrogates)

    return out
