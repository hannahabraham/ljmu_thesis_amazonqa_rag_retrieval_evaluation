"""Build the knowledge base from the 200 sampled records.

The KB stores full review text, not 3 curated chunks. Downstream retrievers
operate over chunked KB rows; Parent-Child uses these full reviews as parents.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.utils.io import parse_list_field

logger = logging.getLogger(__name__)

REVIEW_FIELDS = (
    "review_snippets",
    "top_sentences_IR",
    "top_review_helpful",
    "top_review_wilson",
)
MIN_REVIEW_TOKENS = 5


def _extract_text(item: object) -> str:
    """Some review fields hold strings, some hold dicts with a 'text' key."""
    if item is None:
        return ""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("text", "review_text", "snippet", "sentence"):
            if key in item and item[key]:
                return str(item[key])
        return ""
    return str(item)


def build_knowledge_base(final_records: pd.DataFrame) -> pd.DataFrame:
    """Pull all reviews per record, dedupe, return a tidy KB DataFrame."""
    rows: list[dict] = []
    doc_counter = 1

    for _, row in final_records.iterrows():
        seen: set[str] = set()
        for field in REVIEW_FIELDS:
            for raw in parse_list_field(row.get(field)):
                review_text = _extract_text(raw).strip()
                if len(review_text.split()) < MIN_REVIEW_TOKENS:
                    continue
                norm = " ".join(review_text.lower().split())
                if norm in seen:
                    continue
                seen.add(norm)
                rows.append({
                    "doc_id": f"KB_{doc_counter:05d}",
                    "record_id": row["record_id"],
                    "qid": row["qid"],
                    "asin": row["asin"],
                    "category": row.get("category", "unknown"),
                    "source_file": row["source_file"],
                    "review_text": review_text,
                    "n_words": len(review_text.split()),
                    "source_field": field,
                })
                doc_counter += 1

    kb_df = pd.DataFrame(rows)
    logger.info("Built KB with %d rows from %d records", len(kb_df), len(final_records))
    return kb_df
