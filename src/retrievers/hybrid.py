"""Hybrid retriever using BM25, dense retrieval, and Reciprocal Rank Fusion."""

from __future__ import annotations

import logging
from typing import Any, Sequence

from config.settings import RRF_K
from src.retrievers.base import Retriever
from src.retrievers.bm25 import BM25Retriever
from src.retrievers.dense import DenseRetriever

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k_rrf: int = RRF_K,
) -> list[tuple[str, float]]:
    """Fuse ranked chunk IDs using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}

    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (
                1.0 / (k_rrf + rank)
            )

    return sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )


class HybridRetriever(Retriever):
    """Hybrid retriever that fuses BM25 and dense results."""

    name = "hybrid"

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        retrieve_pool: int = 20,
        k_rrf: int = RRF_K,
    ) -> None:
        """Initialize hybrid retriever."""
        self._bm25 = bm25
        self._dense = dense
        self._pool = retrieve_pool
        self._k_rrf = k_rrf

    @staticmethod
    def build_lookup(
        hits: Sequence[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Build chunk_id lookup from retrieval hits."""
        return {
            str(hit["chunk_id"]): hit
            for hit in hits
            if hit.get("chunk_id")
        }

    @staticmethod
    def chunk_ranking(
        hits: Sequence[dict[str, Any]],
    ) -> list[str]:
        """Extract ranked chunk IDs."""
        return [
            str(hit["chunk_id"])
            for hit in hits
            if hit.get("chunk_id")
        ]

    @staticmethod
    def contributors_for_chunk(
        chunk_id: str,
        bm25_lookup: dict[str, dict[str, Any]],
        dense_lookup: dict[str, dict[str, Any]],
    ) -> list[str]:
        """Return retrievers that contributed a chunk."""
        contributors: list[str] = []

        if chunk_id in bm25_lookup:
            contributors.append("bm25")

        if chunk_id in dense_lookup:
            contributors.append("dense")

        return contributors

    @staticmethod
    def log_source_skew(
        asin: str,
        fused_chunk_ids: list[str],
        bm25_lookup: dict[str, dict[str, Any]],
        dense_lookup: dict[str, dict[str, Any]],
    ) -> None:
        """Log if hybrid results are dominated by one retriever."""
        bm25_only = sum(
            chunk_id in bm25_lookup and chunk_id not in dense_lookup
            for chunk_id in fused_chunk_ids
        )

        dense_only = sum(
            chunk_id in dense_lookup and chunk_id not in bm25_lookup
            for chunk_id in fused_chunk_ids
        )

        total_unique = bm25_only + dense_only

        if total_unique == 0:
            return

        skew_ratio = max(bm25_only, dense_only) / total_unique

        if skew_ratio > 0.8:
            logger.info(
                "Hybrid source skew for asin=%s | bm25_only=%d | dense_only=%d",
                asin,
                bm25_only,
                dense_only,
            )

    def retrieve(
        self,
        question: str,
        asin: str,
        k: int,
    ) -> Sequence[dict[str, Any]]:
        """Return top-k hybrid retrieval results."""
        if k <= 0:
            return []

        bm25_hits = list(
            self._bm25.retrieve(
                question=question,
                asin=asin,
                k=self._pool,
            )
        )

        dense_hits = list(
            self._dense.retrieve(
                question=question,
                asin=asin,
                k=self._pool,
            )
        )

        bm25_lookup = self.build_lookup(bm25_hits)
        dense_lookup = self.build_lookup(dense_hits)

        fused = reciprocal_rank_fusion(
            rankings=[
                self.chunk_ranking(bm25_hits),
                self.chunk_ranking(dense_hits),
            ],
            k_rrf=self._k_rrf,
        )

        fused_chunk_ids = [
            chunk_id for chunk_id, _ in fused
        ]

        self.log_source_skew(
            asin=asin,
            fused_chunk_ids=fused_chunk_ids,
            bm25_lookup=bm25_lookup,
            dense_lookup=dense_lookup,
        )

        results: list[dict[str, Any]] = []

        for chunk_id, fused_score in fused[:k]:
            base_hit = bm25_lookup.get(chunk_id) or dense_lookup.get(chunk_id)

            if base_hit is None:
                continue

            result = dict(base_hit)
            result["score"] = float(fused_score)
            result["retriever"] = self.name
            result["contributors"] = self.contributors_for_chunk(
                chunk_id=chunk_id,
                bm25_lookup=bm25_lookup,
                dense_lookup=dense_lookup,
            )

            if chunk_id in bm25_lookup:
                result["bm25_score"] = bm25_lookup[chunk_id].get("score")

            if chunk_id in dense_lookup:
                result["dense_score"] = dense_lookup[chunk_id].get("score")

            results.append(result)

        return results