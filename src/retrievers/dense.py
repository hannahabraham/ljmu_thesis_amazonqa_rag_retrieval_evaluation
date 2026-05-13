"""Dense retriever using Qdrant and MiniLM embeddings."""

from __future__ import annotations

from typing import Any, Sequence

from config.settings import QDRANT_COLLECTIONS
from src.indexing import get_embedder, get_qdrant_client
from src.retrievers.base import Retriever


class DenseRetriever(Retriever):
    """Dense vector retriever over a Qdrant collection."""

    name = "dense"

    def __init__(
        self,
        collection: str = QDRANT_COLLECTIONS["passages"],
        client: Any | None = None,
        embedder: Any | None = None,
    ) -> None:
        """Initialize dense retriever."""
        self._collection = collection
        self._client = client or get_qdrant_client()
        self._embedder = embedder or get_embedder()

    def embed_query(self, question: str) -> list[float]:
        """Embed query text into a normalized vector."""
        vector = self._embedder.encode(
            [question],
            normalize_embeddings=True,
        )[0]

        return vector.tolist()

    def build_asin_filter(self, asin: str) -> Any:
        """Build Qdrant filter for ASIN."""
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
        """Return top-k ASIN-filtered dense retrieval results."""
        if k <= 0:
            return []

        response = self._client.query_points(
            collection_name=self._collection,
            query=self.embed_query(question),
            query_filter=self.build_asin_filter(asin),
            limit=k,
            with_payload=True,
        )

        results: list[dict[str, Any]] = []

        for hit in response.points:
            payload = hit.payload or {}

            results.append(
                {
                    "chunk_id": payload.get("chunk_id"),
                    "doc_id": payload.get("doc_id"),
                    "record_id": payload.get("record_id"),
                    "asin": payload.get("asin"),
                    "category": payload.get("category"),
                    "text": payload.get("text", ""),
                    "score": float(hit.score),
                    "retriever": self.name,
                }
            )

        return results