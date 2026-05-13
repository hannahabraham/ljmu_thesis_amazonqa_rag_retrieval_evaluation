"""Build and validate the golden answer dataset."""
from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from src.llm_clients.gemini_key_manager import GeminiKeyManager
from src.utils.io import parse_list_field

logger = logging.getLogger(__name__)

JEFFREYS_TIE_DELTA = 0.05
GROUNDING_JACCARD_THRESHOLD = 0.1

JUDGE_PROMPT = """You are evaluating candidate answers to an Amazon product question.
Your task is to pick the single best answer that is GROUNDED in the review evidence,
or to mark the question as unanswerable if no review supports any candidate.

Question: {question}
Product category: {category}
Question type: {question_type}

Candidate answers (with vote counts):
{candidates_block}

Available review evidence:
{evidence_block}

Rules:
1. The chosen answer MUST be supported by at least one review.
2. If no candidate is supported by the reviews, set golden_answer to "[UNANSWERABLE]"
   and evidence_doc_id to null.
3. judge_confidence reflects how clearly the evidence supports the choice
   (1.0 = unambiguous, 0.5 = plausible, 0.0 = guess).

Respond with ONLY a JSON object - no preamble, no markdown fencing - matching this schema:
{{
  "golden_answer": "<chosen answer text or [UNANSWERABLE]>",
  "evidence_doc_id": "<KB_NNNNN of supporting review, or null>",
  "judge_confidence": <float in [0.0, 1.0]>,
  "reasoning": "<1-2 sentence justification>"
}}
"""

CANDIDATE_LINE = "[{idx}] (helpful={helpful}/{total}) {text}"
EVIDENCE_LINE = "[{doc_id}] {review_text}"


# ---------- Jeffreys + grounding heuristics ----------


def jeffreys_score(helpful: int, total: int) -> float:
    return (helpful + 0.5) / (total + 1.0)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def is_grounded(answer_text: str, kb_review_texts: list[str]) -> tuple[bool, str | None, float]:
    """Return (grounded, best_doc_id_index, best_jaccard)."""
    best = 0.0
    best_idx = -1
    for i, review in enumerate(kb_review_texts):
        score = jaccard(answer_text, review)
        if score > best:
            best, best_idx = score, i
    return best >= GROUNDING_JACCARD_THRESHOLD, (str(best_idx) if best_idx >= 0 else None), best


def _vote_counts(answer: dict) -> tuple[int, int]:
    """Return (helpful_votes, total_votes) from an AmazonQA answer dict.

    The raw `helpful` field is a 2-element array [helpful, total]. Older / partial
    records may store separate `helpful` / `unhelpful` scalars instead.
    """
    helpful_field = answer.get("helpful", 0)
    if hasattr(helpful_field, "__len__") and not isinstance(helpful_field, (str, bytes)):
        seq = list(helpful_field)
        if len(seq) >= 2:
            return int(seq[0]), int(seq[1])
        if len(seq) == 1:
            return int(seq[0]), int(seq[0])
        return 0, 0
    helpful = int(helpful_field) if helpful_field is not None else 0
    unhelpful = int(answer.get("unhelpful", 0) or 0)
    return helpful, helpful + unhelpful


def score_candidates(answers: list[dict]) -> list[dict]:
    """Attach Jeffreys score + helpful/total counts to each candidate."""
    scored: list[dict] = []
    for a in answers:
        if not isinstance(a, dict):
            continue
        helpful, total = _vote_counts(a)
        unhelpful = max(0, total - helpful)
        scored.append({
            "text": str(a.get("answerText", a.get("text", ""))),
            "helpful": helpful,
            "unhelpful": unhelpful,
            "total": total,
            "jeffreys": jeffreys_score(helpful, total),
        })
    scored.sort(key=lambda x: -x["jeffreys"])
    return scored


def needs_judge_flag(scored: list[dict], top_grounded: bool, is_answerable: int | None) -> bool:
    """Flag rows where heuristics tie or fail."""
    if not scored:
        return True
    if all(c["total"] == 0 for c in scored):
        return True
    if not top_grounded:
        return True
    if len(scored) >= 2 and abs(scored[0]["jeffreys"] - scored[1]["jeffreys"]) < JEFFREYS_TIE_DELTA:
        return True
    if is_answerable is None:
        return True
    return False


def build_draft_row(
    record_row: pd.Series,
    kb_for_record: pd.DataFrame,
) -> dict[str, Any]:
    """Compute draft golden row for one record."""
    answers = parse_list_field(record_row.get("answers"))
    scored = score_candidates(answers)

    kb_texts = kb_for_record["review_text"].tolist()
    kb_doc_ids = kb_for_record["doc_id"].tolist()

    top_grounded = False
    evidence_doc_id: str | None = None
    evidence_text: str | None = None
    if scored:
        grounded, best_idx_str, _ = is_grounded(scored[0]["text"], kb_texts)
        top_grounded = grounded
        if grounded and best_idx_str is not None:
            idx = int(best_idx_str)
            evidence_doc_id = kb_doc_ids[idx]
            evidence_text = kb_texts[idx]

    flag = needs_judge_flag(scored, top_grounded, record_row.get("is_answerable"))
    selection_method = "grounded_pick" if top_grounded else "jeffreys"

    golden_answer = scored[0]["text"] if scored else ""

    return {
        "record_id": record_row["record_id"],
        "qid": record_row["qid"],
        "asin": record_row["asin"],
        "category": record_row.get("category", "unknown"),
        "source_file": record_row["source_file"],
        "question": record_row["questionText"],
        "question_type": record_row.get("questionType", "unknown"),
        "answerability": record_row.get("is_answerable"),
        "candidate_answers": scored,
        "golden_answer": golden_answer,
        "evidence_text": evidence_text,
        "evidence_doc_id": evidence_doc_id,
        "selection_method": selection_method,
        "needs_judge": flag,
    }


def build_golden_draft(
    final_records: pd.DataFrame,
    kb_df: pd.DataFrame,
) -> pd.DataFrame:
    """Three-layer draft: Jeffreys -> grounding filter -> judge flag."""
    rows: list[dict[str, Any]] = []
    for idx, (_, record) in enumerate(final_records.iterrows(), start=1):
        kb_for_record = kb_df[kb_df["record_id"] == record["record_id"]]
        row = build_draft_row(record, kb_for_record)
        row["golden_id"] = f"G_{idx:03d}"
        rows.append(row)
    return pd.DataFrame(rows)


# ---------- Gemini judge ----------


class JudgeResponse(BaseModel):
    golden_answer: str
    evidence_doc_id: str | None
    judge_confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


def parse_judge_response(raw: str) -> JudgeResponse:
    cleaned = raw.strip()
    cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return JudgeResponse.model_validate_json(cleaned)


def judge_with_retry(
    gemini: GeminiKeyManager,
    prompt: str,
    max_attempts: int = 2,
) -> JudgeResponse:
    last_error: Exception | None = None
    current_prompt = prompt
    for _attempt in range(max_attempts):
        raw = gemini.invoke(current_prompt)
        try:
            return parse_judge_response(raw)
        except (ValidationError, ValueError) as error:
            last_error = error
            current_prompt = (
                f"{prompt}\n\n"
                f"Your previous response was invalid: {error}\n"
                f"Reply with valid JSON only, matching the schema exactly."
            )
    raise RuntimeError(f"Judge failed after {max_attempts} attempts: {last_error}")


def format_judge_prompt(draft_row: dict[str, Any], kb_for_record: pd.DataFrame) -> str:
    candidates_block = "\n".join(
        CANDIDATE_LINE.format(idx=i, helpful=c["helpful"], total=c["total"], text=c["text"])
        for i, c in enumerate(draft_row["candidate_answers"], start=1)
    ) or "(no candidate answers)"

    evidence_block = "\n".join(
        EVIDENCE_LINE.format(doc_id=row["doc_id"], review_text=row["review_text"])
        for _, row in kb_for_record.iterrows()
    ) or "(no review evidence available)"

    return JUDGE_PROMPT.format(
        question=draft_row["question"],
        category=draft_row["category"],
        question_type=draft_row["question_type"],
        candidates_block=candidates_block,
        evidence_block=evidence_block,
    )


# ---------- Validation ----------


def validate_golden_consistency(golden_df: pd.DataFrame, kb_df: pd.DataFrame) -> None:
    """Raise ValueError if golden rows reference KB rows that don't exist or have drifted text."""
    kb_lookup = kb_df.set_index("doc_id")["review_text"].to_dict()
    issues: list[tuple[str, str]] = []

    for _, row in golden_df.iterrows():
        gid = row["golden_id"]
        if row["golden_answer"] == "[UNANSWERABLE]":
            if pd.notna(row["evidence_doc_id"]):
                issues.append((gid, "unanswerable row has non-null evidence_doc_id"))
            continue

        doc_id = row["evidence_doc_id"]
        if pd.isna(doc_id):
            issues.append((gid, "answerable row has null evidence_doc_id"))
            continue

        kb_text = kb_lookup.get(doc_id)
        if kb_text is None:
            issues.append((gid, f"evidence_doc_id={doc_id} not in KB"))
            continue

        if str(row["evidence_text"]).strip() != kb_text.strip():
            issues.append((gid, f"evidence_text drifted from KB[{doc_id}]"))

    if issues:
        raise ValueError(f"Golden dataset inconsistent: {len(issues)} issues -- {issues[:5]}")
