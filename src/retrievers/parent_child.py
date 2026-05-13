"""Parent-child retriever.

Search child chunks, map hits to parent reviews, deduplicate parents,
and return full parent text for generation.
"""

from __future__ import annotations

from typing import Any, Sequence

from config.settings import QDRANT_COLLECTIONS
from src.indexing import get_embedder, get_qdrant_client
from src.retrievers.base import Retriever


class ParentChildRetriever(Retriever):
    """Retrieve child chunks but return full parent review text."""

    name = "pc"

    def __init__(
        self,
        collection: str = QDRANT_COLLECTIONS["child_chunks"],
        client: Any | None = None,
        embedder: Any | None = None,
        child_pool_multiplier: int = 3,
    ) -> None:
        """Initialize parent-child retriever."""
        self._collection = collection
        self._client = client or get_qdrant_client()
        self._embedder = embedder or get_embedder()
        self._pool_multiplier = child_pool_multiplier

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

    def retrieve(
        self,
        question: str,
        asin: str,
        k: int,
    ) -> Sequence[dict[str, Any]]:
        """Return top-k parent reviews using child-chunk retrieval."""
        if k <= 0:
            return []

        response = self._client.query_points(
            collection_name=self._collection,
            query=self.embed_query(question),
            query_filter=self.build_asin_filter(asin),
            limit=k * self._pool_multiplier,
            with_payload=True,
        )

        seen_parent_ids: set[str] = set()
        results: list[dict[str, Any]] = []

        for hit in response.points:
            payload = hit.payload or {}
            parent_id = payload.get("parent_id")

            if parent_id is None:
                continue

            parent_id = str(parent_id)

            if parent_id in seen_parent_ids:
                continue

            seen_parent_ids.add(parent_id)

            results.append(
                {
                    "chunk_id": payload.get("chunk_id"),
                    "child_chunk_id": payload.get("chunk_id"),
                    "doc_id": parent_id,
                    "parent_id": parent_id,
                    "record_id": payload.get("record_id"),
                    "asin": payload.get("asin"),
                    "category": payload.get("category"),
                    "text": payload.get("parent_text", ""),
                    "child_text": payload.get("text", ""),
                    "score": float(hit.score),
                    "retriever": self.name,
                }
            )

            if len(results) >= k:
                break

        return results