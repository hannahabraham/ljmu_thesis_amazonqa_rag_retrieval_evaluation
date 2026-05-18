"""Build RAGAS judge components using Gemini and HuggingFace embeddings."""

from __future__ import annotations

import logging
from typing import Any

from config.settings import EMBEDDING_MODEL, GEMINI_JUDGE_MODEL
from src.llm_clients.loader import load_gemini_keys
from src.llm_clients.round_robin_gemini import RotatingGeminiChat

LOGGER = logging.getLogger(__name__)


def build_ragas_judge() -> tuple[Any, Any, int]:
    """Build RAGAS judge LLM, embeddings, and worker count."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    gemini_keys = load_gemini_keys()
    LOGGER.info(
        "RAGAS judge will rotate across %d Gemini key(s)",
        len(gemini_keys),
    )

    judge_llm = LangchainLLMWrapper(
        RotatingGeminiChat(
            keys=gemini_keys,
            model=GEMINI_JUDGE_MODEL,
            temperature=0.0,
            interactive_replacement=False,
        )
    )

    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
        )
    )

    # The wrapper rotates through keys serially; keep the default worker count
    # conservative so quota errors on one key don't immediately exhaust the
    # rotation. Callers can opt into more workers from the CLI.
    worker_count = 1

    return judge_llm, embeddings, worker_count
