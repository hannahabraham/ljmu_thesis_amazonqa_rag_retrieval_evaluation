"""Sentence-window retriever.

Search sentence chunks, then expand each hit to previous + matched + next
sentence within the same parent review.
"""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from config.settings import QDRANT_COLLECTIONS
from src.indexing import get_embedder, get_qdrant_client
from src.retrievers.base import Retriever


class SentenceWindowRetriever(Retriever):
    """Retrieve matching sentences and return small sentence windows."""

    name = "sentwin"

    def __init__(
        self,
        sentence_chunks: pd.DataFrame,
        collection: str = QDRANT_COLLECTIONS["sentences"],
        client: Any | None = None,
        embedder: Any | None = None,
    ) -> None:
        """Initialize sentence-window retriever."""
        self._collection = collection
        self._client = client or get_qdrant_client()
        self._embedder = embedder or get_embedder()

        self._sentences = sentence_chunks.set_index(
            "chunk_id",
            drop=False,
        )

    def embed_query(self, question: str) -> list[float]:
        """Embed query text into a normalized vector."""
        vector = self._embedder.encode(
            [question],
            normalize_embeddings=True,
        )[0]

        return vector.tolist()

    @staticmethod
    def build_asin_filter(asin: str) -> Any:
        """Build Qdrant ASIN filter."""
        from qdrant_client import models as qm

        return qm.Filter(
            must=[
                qm.FieldCondition(
                    key="asin",
                    match=qm.MatchValue(value=str(asin)),
                )
            ]
        )

    def get_window_ids(self, chunk_id: str) -> list[str]:
        """Return previous, current, and next sentence IDs."""
        row = self._sentences.loc[chunk_id]

        candidate_ids = [
            row.get("prev_sent_id"),
            chunk_id,
            row.get("next_sent_id"),
        ]

        window_ids: list[str] = []

        for sentence_id in candidate_ids:
            if sentence_id is None or pd.isna(sentence_id):
                continue

            sentence_id = str(sentence_id)

            if sentence_id in self._sentences.index:
                window_ids.append(sentence_id)

        return window_ids

    def build_window_result(
        self,
        *,
        chunk_id: str,
        payload: dict[str, Any],
        score: float,
    ) -> dict[str, Any] | None:
        """Build one expanded sentence-window result."""
        if chunk_id not in self._sentences.index:
            return None

        center_row = self._sentences.loc[chunk_id]
        window_ids = self.get_window_ids(chunk_id)

        if not window_ids:
            return None

        window_texts = [
            str(self._sentences.loc[sentence_id, "text"])
            for sentence_id in window_ids
        ]

        indices = [
            int(self._sentences.loc[sentence_id, "sentence_index"])
            for sentence_id in window_ids
        ]

        return {
            "chunk_id": chunk_id,
            "doc_id": center_row.get("doc_id"),
            "record_id": center_row.get("record_id"),
            "asin": payload.get("asin"),
            "category": center_row.get("category"),
            "text": " ".join(window_texts),
            "score": float(score),
            "retriever": self.name,
            "window_chunk_ids": window_ids,
            "window_start_index": min(indices),
            "window_end_index": max(indices),
            "matched_sentence_index": int(center_row["sentence_index"]),
        }

    @staticmethod
    def window_key(result: dict[str, Any]) -> tuple[str, int, int]:
        """Return key used to deduplicate overlapping windows."""
        return (
            str(result.get("doc_id")),
            int(result.get("window_start_index", -1)),
            int(result.get("window_end_index", -1)),
        )

    def retrieve(
        self,
        question: str,
        asin: str,
        k: int,
    ) -> Sequence[dict[str, Any]]:
        """Return top-k ASIN-filtered sentence-window results."""
        if k <= 0:
            return []

        response = self._client.query_points(
            collection_name=self._collection,
            query=self.embed_query(question),
            query_filter=self.build_asin_filter(asin),
            limit=k * 2,
            with_payload=True,
        )

        seen_windows: set[tuple[str, int, int]] = set()
        results: list[dict[str, Any]] = []

        for hit in response.points:
            payload = hit.payload or {}
            chunk_id = payload.get("chunk_id")

            if chunk_id is None:
                continue

            result = self.build_window_result(
                chunk_id=str(chunk_id),
                payload=payload,
                score=float(hit.score),
            )

            if result is None:
                continue

            key = self.window_key(result)

            if key in seen_windows:
                continue

            seen_windows.add(key)
            results.append(result)

            if len(results) >= k:
                break

        return results
