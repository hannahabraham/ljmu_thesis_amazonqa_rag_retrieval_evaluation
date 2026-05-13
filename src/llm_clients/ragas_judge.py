"""Build RAGAS judge components using Gemini and HuggingFace embeddings."""

from __future__ import annotations

from typing import Any

from config.settings import EMBEDDING_MODEL, GEMINI_JUDGE_MODEL
from src.llm_clients.loader import load_gemini_keys
from src.llm_clients.round_robin_gemini import RoundRobinGeminiChat


def build_ragas_judge() -> tuple[Any, Any, int]:
    """Build RAGAS judge LLM, embeddings, and worker count."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    gemini_keys = load_gemini_keys()

    judge_llm = LangchainLLMWrapper(
        RoundRobinGeminiChat(
            keys=gemini_keys,
            model=GEMINI_JUDGE_MODEL,
            temperature=0.0,
        )
    )

    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
        )
    )

    worker_count = len(gemini_keys)

    return judge_llm, embeddings, worker_count