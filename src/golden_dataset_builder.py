"""Build and validate the golden answer dataset."""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from src.llm_clients.gemini_key_manager import GeminiKeyManager
from src.utils.io import parse_list_field

LOGGER = logging.getLogger(__name__)

JEFFREYS_TIE_DELTA = 0.05
GROUNDING_JACCARD_THRESHOLD = 0.1

# Rows whose top candidate's Jaccard to the best KB review falls below this
# (stricter) bar still go to the judge even when nominally "grounded",
# because token Jaccard at 0.1 misses ungrounded yes/no answers like "yes"
# that incidentally share stopwords with a review.
JUDGE_REVIEW_JACCARD_THRESHOLD = 0.2

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


def jeffreys_score(helpful: int, total: int) -> float:
    """Return Jeffreys-smoothed helpfulness score."""
    return (helpful + 0.5) / (total + 1.0)


def _tokens(text: str) -> set[str]:
    """Tokenize text into lowercase alphanumeric terms."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(first_text: str, second_text: str) -> float:
    """Return Jaccard similarity between two texts."""
    first_tokens = _tokens(first_text)
    second_tokens = _tokens(second_text)

    if not first_tokens or not second_tokens:
        return 0.0

    return len(first_tokens & second_tokens) / len(
        first_tokens | second_tokens
    )


def is_grounded(
    answer_text: str,
    kb_review_texts: list[str],
) -> tuple[bool, str | None, float]:
    """Return whether an answer is grounded in the provided KB reviews."""
    best_score = 0.0
    best_index = -1

    for index, review_text in enumerate(kb_review_texts):
        score = jaccard(answer_text, review_text)

        if score > best_score:
            best_score = score
            best_index = index

    return (
        best_score >= GROUNDING_JACCARD_THRESHOLD,
        str(best_index) if best_index >= 0 else None,
        best_score,
    )


def _vote_counts(answer: dict[str, Any]) -> tuple[int, int]:
    """Return helpful and total vote counts from an AmazonQA answer."""
    helpful_field = answer.get("helpful", 0)

    if hasattr(helpful_field, "__len__") and not isinstance(
        helpful_field,
        (str, bytes),
    ):
        sequence = list(helpful_field)

        if len(sequence) >= 2:
            return int(sequence[0]), int(sequence[1])

        if len(sequence) == 1:
            return int(sequence[0]), int(sequence[0])

        return 0, 0

    helpful = int(helpful_field) if helpful_field is not None else 0
    unhelpful = int(answer.get("unhelpful", 0) or 0)

    return helpful, helpful + unhelpful


def score_candidates(answers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach Jeffreys score and vote counts to candidate answers."""
    scored: list[dict[str, Any]] = []

    for answer in answers:
        if not isinstance(answer, dict):
            continue

        helpful, total = _vote_counts(answer)
        unhelpful = max(0, total - helpful)

        scored.append(
            {
                "text": str(answer.get("answerText", answer.get("text", ""))),
                "helpful": helpful,
                "unhelpful": unhelpful,
                "total": total,
                "jeffreys": jeffreys_score(helpful, total),
            }
        )

    scored.sort(key=lambda candidate: -candidate["jeffreys"])

    return scored


def needs_judge_flag(
    scored: list[dict[str, Any]],
    top_grounded: bool,
    is_answerable: int | None,
    *,
    question_type: str | None = None,
    top_grounding_score: float | None = None,
) -> bool:
    """Return whether a row should be sent to the LLM judge.

    Aggressive gating (supervisor brief 2026-05-16): we route to the judge
    whenever any of the following is true. The judge is cheap on Gemini
    Flash; the cost of a wrong heuristic-pick is much higher.

    Triggers:
      * no candidate answers at all
      * every candidate has zero helpful votes
      * top candidate is not grounded in the KB (Jaccard < 0.1)
      * top candidate is grounded but only weakly (Jaccard < 0.2) - the
        0.1 threshold is too loose for short answers like "yes"/"no"
      * Jeffreys score is tied within ``JEFFREYS_TIE_DELTA``
      * answerability is missing or marked unanswerable (``is_answerable``
        in ``{None, 0}``) - heuristics can't confirm "no answer in reviews"
      * question type is "yesno" - heuristic picks are weakest here
        because vote distributions for short answers are noisy
    """
    if not scored:
        return True

    if all(candidate["total"] == 0 for candidate in scored):
        return True

    if not top_grounded:
        return True

    if (
        top_grounding_score is not None
        and top_grounding_score < JUDGE_REVIEW_JACCARD_THRESHOLD
    ):
        return True

    if (
        len(scored) >= 2
        and abs(scored[0]["jeffreys"] - scored[1]["jeffreys"])
        < JEFFREYS_TIE_DELTA
    ):
        return True

    if is_answerable is None or is_answerable == 0:
        return True

    if question_type is not None and str(question_type).strip().lower() == "yesno":
        return True

    return False


def build_draft_row(
    record_row: pd.Series,
    kb_for_record: pd.DataFrame,
) -> dict[str, Any]:
    """Build one draft golden dataset row."""
    answers = parse_list_field(record_row.get("answers"))
    scored = score_candidates(answers)

    kb_texts = kb_for_record["review_text"].tolist()
    kb_doc_ids = kb_for_record["doc_id"].tolist()

    top_grounded = False
    top_grounding_score = 0.0
    evidence_doc_id: str | None = None
    evidence_text: str | None = None

    if scored:
        grounded, best_index_text, jaccard_score = is_grounded(
            scored[0]["text"],
            kb_texts,
        )
        top_grounded = grounded
        top_grounding_score = jaccard_score

        if grounded and best_index_text is not None:
            best_index = int(best_index_text)
            evidence_doc_id = kb_doc_ids[best_index]
            evidence_text = kb_texts[best_index]

    needs_judge = needs_judge_flag(
        scored,
        top_grounded,
        record_row.get("is_answerable"),
        question_type=record_row.get("questionType"),
        top_grounding_score=top_grounding_score,
    )

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
        "needs_judge": needs_judge,
    }


def build_golden_draft(
    final_records: pd.DataFrame,
    kb_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the draft golden dataset using heuristics and judge flags."""
    rows: list[dict[str, Any]] = []

    for index, (_, record) in enumerate(
        final_records.iterrows(),
        start=1,
    ):
        kb_for_record = kb_df[
            kb_df["record_id"] == record["record_id"]
        ]

        row = build_draft_row(
            record,
            kb_for_record,
        )
        row["golden_id"] = f"G_{index:03d}"

        rows.append(row)

    return pd.DataFrame(rows)


class JudgeResponse(BaseModel):
    """Validated JSON schema for Gemini judge responses."""

    golden_answer: str
    evidence_doc_id: str | None
    judge_confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


def parse_judge_response(raw: str) -> JudgeResponse:
    """Parse a Gemini judge response into a validated object."""
    cleaned = raw.strip()
    cleaned = (
        cleaned.removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )

    return JudgeResponse.model_validate_json(cleaned)


def judge_with_retry(
    gemini: GeminiKeyManager,
    prompt: str,
    max_attempts: int = 2,
) -> JudgeResponse:
    """Call Gemini judge with one retry prompt on invalid JSON."""
    last_error: Exception | None = None
    current_prompt = prompt

    for _ in range(max_attempts):
        raw_response = gemini.invoke(current_prompt)

        try:
            return parse_judge_response(raw_response)
        except (ValidationError, ValueError) as error:
            last_error = error
            current_prompt = (
                f"{prompt}\n\n"
                f"Your previous response was invalid: {error}\n"
                "Reply with valid JSON only, matching the schema exactly."
            )

    raise RuntimeError(
        f"Judge failed after {max_attempts} attempts: {last_error}"
    )


def format_judge_prompt(
    draft_row: dict[str, Any],
    kb_for_record: pd.DataFrame,
) -> str:
    """Format the Gemini judge prompt for one draft golden row."""
    candidates_block = "\n".join(
        CANDIDATE_LINE.format(
            idx=index,
            helpful=candidate["helpful"],
            total=candidate["total"],
            text=candidate["text"],
        )
        for index, candidate in enumerate(
            draft_row["candidate_answers"],
            start=1,
        )
    ) or "(no candidate answers)"

    evidence_block = "\n".join(
        EVIDENCE_LINE.format(
            doc_id=row["doc_id"],
            review_text=row["review_text"],
        )
        for _, row in kb_for_record.iterrows()
    ) or "(no review evidence available)"

    return JUDGE_PROMPT.format(
        question=draft_row["question"],
        category=draft_row["category"],
        question_type=draft_row["question_type"],
        candidates_block=candidates_block,
        evidence_block=evidence_block,
    )


def validate_golden_consistency(
    golden_df: pd.DataFrame,
    kb_df: pd.DataFrame,
) -> pd.DataFrame:
    """Validate golden evidence references against the KB.

    Two severities:

    Hard failures (raise ``ValueError`` at the end if any are found):
      * ``answerability`` column is missing on a row.
      * answerable row has null ``evidence_doc_id``.
      * ``evidence_doc_id`` does not exist in the KB.
      * ``evidence_text`` has drifted from the KB record verbatim.

    Soft warnings (logged + recorded in ``validation_status`` on the
    returned frame, no exception raised):
      * ``golden_answer == "[UNANSWERABLE]"`` but ``answerability == 1``,
        or vice versa. The advisory AmazonQA label and the judge/grounding
        outcome can legitimately disagree (see ``_normalise_answerability``
        in step08a) — flag for review rather than failing the dataset.
      * unanswerable row carries a non-null ``evidence_doc_id``.

    Returns:
        A copy of ``golden_df`` with a ``validation_status`` column
        (``"ok"``, ``"label_mismatch"``, ``"unanswerable_with_evidence"``,
        ``"multiple_warnings"``).

    """
    kb_lookup = kb_df.set_index("doc_id")["review_text"].to_dict()
    hard_issues: list[tuple[str, str]] = []
    statuses: list[str] = []

    for _, row in golden_df.iterrows():
        golden_id = row["golden_id"]
        row_warnings: list[str] = []

        is_unanswerable = (
            str(row["golden_answer"]).strip().upper() == "[UNANSWERABLE]"
        )
        answerability = row.get("answerability")
        if pd.isna(answerability):
            hard_issues.append((golden_id, "answerability is missing"))
            statuses.append("hard_failure")
            continue
        answerability_flag = int(answerability)

        if is_unanswerable and answerability_flag != 0:
            row_warnings.append("label_mismatch")
        elif (not is_unanswerable) and answerability_flag != 1:
            row_warnings.append("label_mismatch")

        if is_unanswerable:
            if pd.notna(row["evidence_doc_id"]):
                row_warnings.append("unanswerable_with_evidence")
        else:
            doc_id = row["evidence_doc_id"]

            if pd.isna(doc_id):
                hard_issues.append(
                    (golden_id, "answerable row has null evidence_doc_id")
                )
                statuses.append("hard_failure")
                continue

            kb_text = kb_lookup.get(doc_id)

            if kb_text is None:
                hard_issues.append(
                    (golden_id, f"evidence_doc_id={doc_id} not in KB")
                )
                statuses.append("hard_failure")
                continue

            if str(row["evidence_text"]).strip() != kb_text.strip():
                hard_issues.append(
                    (golden_id, f"evidence_text drifted from KB[{doc_id}]")
                )
                statuses.append("hard_failure")
                continue

        if not row_warnings:
            statuses.append("ok")
        elif len(row_warnings) == 1:
            statuses.append(row_warnings[0])
        else:
            statuses.append("multiple_warnings")

    annotated = golden_df.copy()
    annotated["validation_status"] = statuses

    soft_counts = (
        annotated["validation_status"]
        .value_counts()
        .to_dict()
    )
    LOGGER.info(
        "Golden validation: %s",
        ", ".join(
            f"{status}={count}" for status, count in soft_counts.items()
        ),
    )

    if hard_issues:
        raise ValueError(
            "Golden dataset has hard integrity failures: "
            f"{len(hard_issues)} issues -- {hard_issues[:5]}"
        )

    return annotated
