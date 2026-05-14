"""Build BM25 and Qdrant retrieval indexes."""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    INDEX_DIR,
    QDRANT_COLLECTIONS,
    QDRANT_HOST,
    QDRANT_PORT,
)

LOGGER = logging.getLogger(__name__)

QDRANT_UPSERT_BATCH_SIZE = 512


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 indexing."""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_bm25_index(
    passage_chunks: pd.DataFrame,
    out_path: Path = INDEX_DIR / "bm25.pkl",
) -> Path:
    """Build and persist a BM25 index for passage chunks."""
    from rank_bm25 import BM25Okapi  # pylint: disable=import-outside-toplevel

    tokenized_corpus = [
        tokenize(text)
        for text in passage_chunks["text"].tolist()
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    payload = {
        "bm25": bm25,
        "chunk_ids": passage_chunks["chunk_id"].tolist(),
        "doc_ids": passage_chunks["doc_id"].tolist(),
        "record_ids": passage_chunks["record_id"].tolist(),
        "asins": passage_chunks["asin"].tolist(),
        "texts": passage_chunks["text"].tolist(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("wb") as file_handle:
        pickle.dump(payload, file_handle)

    LOGGER.info(
        "BM25 index pickled to %s (%d documents)",
        out_path,
        len(tokenized_corpus),
    )

    return out_path


def get_qdrant_client() -> Any:
    """Create and return a Qdrant client."""
    from qdrant_client import QdrantClient  # pylint: disable=import-outside-toplevel

    return QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
    )


def get_embedder() -> Any:
    """Create and return the sentence-transformer embedding model."""
    from sentence_transformers import (  # pylint: disable=import-outside-toplevel
        SentenceTransformer,
    )

    return SentenceTransformer(EMBEDDING_MODEL)


def _embed(
    model: Any,
    texts: list[str],
    batch_size: int = 64,
) -> np.ndarray:
    """Embed texts into normalised dense vectors."""
    return np.asarray(
        model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
    )


def _build_payload(
    row: pd.Series,
    text_col: str,
    extra_payload_cols: tuple[str, ...],
) -> dict[str, Any]:
    """Build a Qdrant payload dictionary from one chunk row."""
    payload = {
        "chunk_id": row["chunk_id"],
        "doc_id": row.get("doc_id"),
        "record_id": row.get("record_id"),
        "asin": row["asin"],
        "category": row.get("category"),
        "text": row[text_col],
    }

    for column_name in extra_payload_cols:
        if column_name in row:
            payload[column_name] = row[column_name]

    return payload


def upsert_collection(
    client: Any,
    collection_name: str,
    chunks_df: pd.DataFrame,
    text_col: str = "text",
    extra_payload_cols: tuple[str, ...] = (),
) -> None:
    """Recreate a Qdrant collection and upsert embedded chunk rows."""
    from qdrant_client import models as qm  # pylint: disable=import-outside-toplevel

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=qm.VectorParams(
            size=EMBEDDING_DIM,
            distance=qm.Distance.COSINE,
        ),
    )

    client.create_payload_index(
        collection_name=collection_name,
        field_name="asin",
        field_schema=qm.PayloadSchemaType.KEYWORD,
    )

    embedder = get_embedder()

    vectors = _embed(
        embedder,
        chunks_df[text_col].tolist(),
    )

    points: list[qm.PointStruct] = []

    for index, (_, row) in enumerate(chunks_df.iterrows()):
        points.append(
            qm.PointStruct(
                id=index,
                vector=vectors[index].tolist(),
                payload=_build_payload(
                    row,
                    text_col,
                    extra_payload_cols,
                ),
            )
        )

    for start_index in range(0, len(points), QDRANT_UPSERT_BATCH_SIZE):
        batch = points[
            start_index:start_index + QDRANT_UPSERT_BATCH_SIZE
        ]

        client.upsert(
            collection_name=collection_name,
            points=batch,
        )

        LOGGER.info(
            "Upserted points %d-%d/%d into collection %r",
            start_index + 1,
            start_index + len(batch),
            len(points),
            collection_name,
        )

    LOGGER.info(
        "Upserted %d points into collection %r",
        len(points),
        collection_name,
    )


def build_all_qdrant_collections(
    passage_chunks: pd.DataFrame,
    sentence_chunks: pd.DataFrame,
    parent_child_chunks: pd.DataFrame,
) -> None:
    """Build all configured Qdrant retrieval collections."""
    client = get_qdrant_client()

    upsert_collection(
        client,
        QDRANT_COLLECTIONS["passages"],
        passage_chunks,
    )

    upsert_collection(
        client,
        QDRANT_COLLECTIONS["sentences"],
        sentence_chunks,
        extra_payload_cols=(
            "sentence_index",
            "prev_sent_id",
            "next_sent_id",
        ),
    )

    upsert_collection(
        client,
        QDRANT_COLLECTIONS["child_chunks"],
        parent_child_chunks,
        extra_payload_cols=(
            "parent_id",
            "parent_text",
        ),
    )