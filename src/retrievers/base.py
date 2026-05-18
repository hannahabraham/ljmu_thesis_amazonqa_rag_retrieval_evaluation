"""Common retriever interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence


class Retriever(ABC):
    """Base interface for all retrieval strategies."""

    name: str

    @abstractmethod
    def retrieve(
        self,
        question: str,
        asin: str,
        k: int,
    ) -> Sequence[dict[str, Any]]:
        """Return top-k ASIN-filtered retrieval results.

        Each result should contain at least:
            chunk_id
            doc_id
            text
            score

        Retrievers may also include extra fields such as:
            parent_id
            parent_text
            sentence_index
            prev_sent_id
            next_sent_id
            contributors
        """
