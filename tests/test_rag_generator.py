"""Unit tests for RAG answer generation utilities."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.generation.rag_generator import format_context, generate_rag_answer


def test_format_context_uses_doc_id_headers() -> None:
    """Test formatted context includes document IDs and review text."""
    documents = [
        {"doc_id": "KB_00001", "text": "first review"},
        {"doc_id": "KB_00002", "text": "second review"},
    ]

    context = format_context(documents)

    assert "[KB_00001]" in context
    assert "[KB_00002]" in context
    assert "first review" in context
    assert "second review" in context


def test_generate_rag_answer_uses_injected_manager() -> None:
    """Test answer generation uses the injected LLM manager."""
    manager = MagicMock()
    manager.invoke.return_value = "Yes, it is waterproof."

    documents = [
        {
            "doc_id": "KB_00001",
            "text": "Waterproof to 10m.",
        },
    ]

    output = generate_rag_answer(
        question="Is it waterproof?",
        retrieved_docs=documents,
        groq_manager=manager,
        retrieval_ms=12.3,
    )

    assert output["generated_answer"] == "Yes, it is waterproof."
    assert output["refused"] is False
    assert output["retrieval_ms"] == 12.3
    assert output["generation_ms"] >= 0

    manager.invoke.assert_called_once()


def test_generate_rag_answer_detects_refusal() -> None:
    """Test refusal detection in generated answers."""
    manager = MagicMock()
    manager.invoke.return_value = "The reviews don't say anything about that."

    output = generate_rag_answer(
        question="Q?",
        retrieved_docs=[
            {
                "doc_id": "X",
                "text": "...",
            },
        ],
        groq_manager=manager,
        retrieval_ms=1.0,
    )

    assert output["refused"] is True