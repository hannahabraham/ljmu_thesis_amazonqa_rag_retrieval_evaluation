"""Resolve judge-flagged rows with Gemini and write verified golden data."""

from __future__ import annotations

import ast
import json
import logging
from typing import Any

import pandas as pd

from config.settings import GEMINI_JUDGE_MODEL, PROCESSED_DIR
from src.golden_dataset_builder import format_judge_prompt, judge_with_retry
from src.llm_clients.gemini_key_manager import GeminiKeyManager
from src.llm_clients.loader import load_gemini_keys
from src.utils.caching import get_cached, set_cached
from src.utils.logging_config import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)


def _parse_candidate_answers(value: Any) -> list[dict[str, Any]]:
    """Parse candidate answers loaded from CSV."""
    if isinstance(value, list):
        return value

    if not isinstance(value, str) or not value.strip():
        return []

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return []

    if not isinstance(parsed, list):
        return []

    return [
        item
        for item in parsed
        if isinstance(item, dict)
    ]


def _row_dict(row: pd.Series) -> dict[str, Any]:
    """Convert a CSV row into a plain dictionary with parsed candidates."""
    record = row.to_dict()
    record["candidate_answers"] = _parse_candidate_answers(
        record.get("candidate_answers")
    )

    return record


def _as_bool(value: Any) -> bool:
    """Parse bool-like values from CSV rows."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _write_partial(
    verified_by_record: dict[str, dict[str, Any]],
    record_order: list[str],
    partial_path: Any,
) -> None:
    """Write processed rows in draft order for interruption recovery."""
    rows = [
        verified_by_record[record_id]
        for record_id in record_order
        if record_id in verified_by_record
    ]

    pd.DataFrame(rows).to_csv(partial_path, index=False)


def _normalize_evidence_doc_id(
    evidence_doc_id: Any,
    evidence_lookup: dict[str, str],
) -> str | None:
    """Return one valid evidence doc id from judge output."""
    if evidence_doc_id is None or pd.isna(evidence_doc_id):
        return None

    for doc_id in str(evidence_doc_id).split(","):
        normalized = doc_id.strip()
        if normalized in evidence_lookup:
            return normalized

    return str(evidence_doc_id).strip()


def _apply_judge_result(
    record: dict[str, Any],
    judged: dict[str, Any],
    evidence_lookup: dict[str, str],
) -> dict[str, Any]:
    """Apply Gemini judge output to a draft golden record."""
    evidence_doc_id = _normalize_evidence_doc_id(
        judged.get("evidence_doc_id"),
        evidence_lookup,
    )

    record.update(
        {
            "golden_answer": judged["golden_answer"],
            "evidence_doc_id": evidence_doc_id,
            "selection_method": "judge",
            "llm_judge_used": True,
            "llm_judge_model": GEMINI_JUDGE_MODEL,
            "judge_confidence": judged["judge_confidence"],
            "verification_status": "judge",
            "evidence_text": (
                evidence_lookup.get(evidence_doc_id)
                if evidence_doc_id is not None
                else None
            ),
        }
    )

    return record


def main() -> None:
    """Resolve flagged rows and save the verified golden dataset."""
    draft_path = PROCESSED_DIR / "golden_dataset_200_draft.csv"
    knowledge_base_path = PROCESSED_DIR / "knowledge_base_full_reviews.csv"
    output_path = PROCESSED_DIR / "golden_dataset_200_verified.csv"
    partial_path = PROCESSED_DIR / "golden_dataset_200_verified.partial.csv"

    draft = pd.read_csv(draft_path)
    knowledge_base = pd.read_csv(knowledge_base_path)
    record_order = draft["record_id"].astype(str).tolist()
    needs_judge_by_record = {
        str(row["record_id"]): _as_bool(row["needs_judge"])
        for _, row in draft.iterrows()
    }
    total_judged = sum(needs_judge_by_record.values())

    evidence_lookup = knowledge_base.set_index("doc_id")[
        "review_text"
    ].to_dict()

    gemini = GeminiKeyManager(
        load_gemini_keys(),
        GEMINI_JUDGE_MODEL,
    )

    verified_by_record: dict[str, dict[str, Any]] = {}
    if partial_path.exists():
        partial = pd.read_csv(partial_path)
        verified_by_record = {
            str(row["record_id"]): row.to_dict()
            for _, row in partial.iterrows()
        }
        LOGGER.info(
            "Resuming from %s with %d/%d rows already processed",
            partial_path,
            len(verified_by_record),
            len(draft),
        )

    judged_done = sum(
        1
        for record_id in verified_by_record
        if needs_judge_by_record.get(record_id, False)
    )

    for _, row in draft.iterrows():
        record = _row_dict(row)
        record_id = str(record["record_id"])
        rows_done = len(verified_by_record)

        if record_id in verified_by_record:
            continue

        if not _as_bool(record["needs_judge"]):
            record.update(
                {
                    "llm_judge_used": False,
                    "llm_judge_model": None,
                    "judge_confidence": None,
                    "verification_status": "heuristic",
                }
            )
            verified_by_record[record_id] = record
            _write_partial(verified_by_record, record_order, partial_path)
            LOGGER.info(
                "Processed row %d/%d without judge (record_id=%s, golden_id=%s)",
                rows_done + 1,
                len(draft),
                record_id,
                record.get("golden_id"),
            )
            continue

        kb_for_record = knowledge_base[
            knowledge_base["record_id"] == record["record_id"]
        ]

        prompt = format_judge_prompt(record, kb_for_record)

        cached_response = get_cached(
            "gemini_judge",
            prompt,
            GEMINI_JUDGE_MODEL,
        )

        if cached_response is not None:
            judged = cached_response
            response_source = "cache"
        else:
            LOGGER.info(
                "Calling Gemini for judge %d/%d (row %d/%d, record_id=%s, golden_id=%s)",
                judged_done + 1,
                total_judged,
                rows_done + 1,
                len(draft),
                record_id,
                record.get("golden_id"),
            )
            response = judge_with_retry(gemini, prompt)
            judged = response.model_dump()
            response_source = "api"

            set_cached(
                "gemini_judge",
                judged,
                prompt,
                GEMINI_JUDGE_MODEL,
            )

        verified_by_record[record_id] = _apply_judge_result(
            record,
            judged,
            evidence_lookup,
        )
        judged_done += 1
        _write_partial(verified_by_record, record_order, partial_path)
        LOGGER.info(
            "Judged %d/%d via %s (row %d/%d, record_id=%s, golden_id=%s)",
            judged_done,
            total_judged,
            response_source,
            rows_done + 1,
            len(draft),
            record_id,
            record.get("golden_id"),
        )

    verified_rows = [
        verified_by_record[record_id]
        for record_id in record_order
    ]
    verified = pd.DataFrame(verified_rows)
    verified.to_csv(output_path, index=False)

    LOGGER.info("Verified golden dataset written to %s", output_path)


if __name__ == "__main__":
    main()
