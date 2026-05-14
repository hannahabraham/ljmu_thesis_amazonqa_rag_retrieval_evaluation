"""Build a full-review knowledge base from sampled AmazonQA records.

The knowledge base stores full review text rather than curated chunks.
Downstream retrievers operate over chunked KB rows, while parent-child
retrieval uses these full reviews as parent documents.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.utils.io import parse_list_field

LOGGER = logging.getLogger(__name__)

REVIEW_FIELDS = (
    "review_snippets",
    "top_sentences_IR",
    "top_review_helpful",
    "top_review_wilson",
)

MIN_REVIEW_TOKENS = 5


def _extract_text(item: Any) -> str:
    """Extract review text from strings or dictionary-like review objects."""
    if item is None:
        return ""

    if isinstance(item, str):
        return item

    if isinstance(item, dict):
        for key in ("text", "review_text", "snippet", "sentence"):
            value = item.get(key)
            if value:
                return str(value)

        return ""

    return str(item)


def build_knowledge_base(final_records: pd.DataFrame) -> pd.DataFrame:
    """Extract, deduplicate, and return review-level KB rows."""
    rows: list[dict[str, Any]] = []
    document_counter = 1

    for _, record in final_records.iterrows():
        seen_reviews: set[str] = set()

        for field_name in REVIEW_FIELDS:
            for raw_review in parse_list_field(record.get(field_name)):
                review_text = _extract_text(raw_review).strip()

                if len(review_text.split()) < MIN_REVIEW_TOKENS:
                    continue

                normalised_review = " ".join(
                    review_text.lower().split()
                )

                if normalised_review in seen_reviews:
                    continue

                seen_reviews.add(normalised_review)

                rows.append(
                    {
                        "doc_id": f"KB_{document_counter:05d}",
                        "record_id": record["record_id"],
                        "qid": record["qid"],
                        "asin": record["asin"],
                        "category": record.get("category", "unknown"),
                        "source_file": record["source_file"],
                        "review_text": review_text,
                        "n_words": len(review_text.split()),
                        "source_field": field_name,
                    }
                )

                document_counter += 1

    knowledge_base = pd.DataFrame(rows)

    LOGGER.info(
        "Built knowledge base with %d rows from %d records",
        len(knowledge_base),
        len(final_records),
    )

    return knowledge_base