"""RAG answer generator. Same template across all pipelines."""
from __future__ import annotations

import time
from typing import Any

from src.generation.prompt import PROMPT_TEMPLATE
from src.generation.refusal import is_refusal


def format_context(retrieved_docs: list[dict]) -> str:
    """Format retrieved docs into the bracketed-id block consumed by PROMPT_TEMPLATE."""
    return "\n\n".join(
        f"[{doc.get('doc_id', doc.get('chunk_id', '?'))}]\n{doc.get('text', '')}"
        for doc in retrieved_docs
    )


def generate_rag_answer(
    question: str,
    retrieved_docs: list[dict],
    groq_manager: Any,
    retrieval_ms: float,
) -> dict:
    """Generate one RAG answer and return generated text, refusal flag, and latencies."""
    prompt = PROMPT_TEMPLATE.format(
        question=question, context=format_context(retrieved_docs),
    )
    start = time.perf_counter()
    answer = groq_manager.invoke(prompt)
    generation_ms = (time.perf_counter() - start) * 1000.0
    return {
        "generated_answer": answer,
        "refused": is_refusal(answer),
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
    }
