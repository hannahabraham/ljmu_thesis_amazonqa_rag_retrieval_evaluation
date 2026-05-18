"""BM25 retriever for keyword-based retrieval."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.indexing import tokenize
from src.retrievers.base import Retriever


class BM25Retriever(Retriever):
    """BM25 keyword retriever over passage chunks."""

    name = "bm25"

    def __init__(
        self,
        chunks: pd.DataFrame,
        text_col: str = "text",
    ) -> None:
        """Initialize BM25 index from a chunk dataframe."""
        from rank_bm25 import BM25Okapi

        if text_col not in chunks.columns:
            raise ValueError(f"Missing text column: {text_col}")

        if "asin" not in chunks.columns:
            raise ValueError("Input dataframe must contain an 'asin' column.")

        self._df = chunks.reset_index(drop=True)
        self._texts = self._df[text_col].fillna("").astype(str).tolist()
        self._asins = self._df["asin"].fillna("").astype(str).tolist()

        tokenized_corpus = [
            tokenize(text)
            for text in self._texts
        ]

        self._bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(
        self,
        question: str,
        asin: str,
        k: int,
    ) -> Sequence[dict[str, Any]]:
        """Return top-k ASIN-filtered BM25 results."""
        if k <= 0:
            return []

        query_tokens = tokenize(question)
        scores = np.asarray(
            self._bm25.get_scores(query_tokens),
            dtype=float,
        )

        asin_mask = np.asarray(
            [value == str(asin) for value in self._asins],
            dtype=bool,
        )

        if not asin_mask.any():
            return []

        masked_scores = np.where(
            asin_mask,
            scores,
            -np.inf,
        )

        top_indices = np.argsort(-masked_scores)[:k]

        results: list[dict[str, Any]] = []

        for index in top_indices:
            score = masked_scores[int(index)]

            if not np.isfinite(score):
                continue

            row = self._df.iloc[int(index)]

            results.append(
                {
                    "chunk_id": row.get("chunk_id"),
                    "doc_id": row.get("doc_id", row.get("chunk_id")),
                    "record_id": row.get("record_id"),
                    "asin": row.get("asin"),
                    "category": row.get("category"),
                    "text": self._texts[int(index)],
                    "score": float(score),
                    "retriever": self.name,
                }
            )

        return results

    def save(self, path: Path) -> None:
        """Persist this retriever to ``path`` (pickle)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "BM25Retriever":
        """Load a previously pickled retriever from ``path``."""
        with open(path, "rb") as handle:
            obj = pickle.load(handle)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} did not contain a {cls.__name__}")
        return obj
