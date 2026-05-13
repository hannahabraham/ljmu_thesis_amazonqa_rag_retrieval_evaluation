"""Build BM25 and Qdrant indexes."""
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

logger = logging.getLogger(__name__)

QDRANT_UPSERT_BATCH_SIZE = 512


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------- BM25 ----------


def build_bm25_index(passage_chunks: pd.DataFrame, out_path: Path = INDEX_DIR / "bm25.pkl") -> Path:
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [tokenize(t) for t in passage_chunks["text"].tolist()]
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
    with out_path.open("wb") as fh:
        pickle.dump(payload, fh)
    logger.info("BM25 index pickled to %s (%d docs)", out_path, len(tokenized_corpus))
    return out_path


# ---------- Qdrant ----------


def get_qdrant_client() -> Any:
    from qdrant_client import QdrantClient

    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def get_embedder() -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def _embed(model: Any, texts: list[str], batch_size: int = 64) -> np.ndarray:
    return np.asarray(
        model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    )


def upsert_collection(
    client: Any,
    collection_name: str,
    chunks_df: pd.DataFrame,
    text_col: str = "text",
    extra_payload_cols: tuple[str, ...] = (),
) -> None:
    from qdrant_client import models as qm

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qm.VectorParams(size=EMBEDDING_DIM, distance=qm.Distance.COSINE),
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="asin",
        field_schema=qm.PayloadSchemaType.KEYWORD,
    )

    embedder = get_embedder()
    vectors = _embed(embedder, chunks_df[text_col].tolist())

    points: list[qm.PointStruct] = []
    for i, (_, row) in enumerate(chunks_df.iterrows()):
        payload = {
            "chunk_id": row["chunk_id"],
            "doc_id": row.get("doc_id"),
            "record_id": row.get("record_id"),
            "asin": row["asin"],
            "category": row.get("category"),
            "text": row[text_col],
        }
        for extra in extra_payload_cols:
            if extra in row:
                payload[extra] = row[extra]
        points.append(qm.PointStruct(id=i, vector=vectors[i].tolist(), payload=payload))

    for start in range(0, len(points), QDRANT_UPSERT_BATCH_SIZE):
        batch = points[start:start + QDRANT_UPSERT_BATCH_SIZE]
        client.upsert(collection_name=collection_name, points=batch)
        logger.info(
            "Upserted points %d-%d/%d into collection %r",
            start + 1,
            start + len(batch),
            len(points),
            collection_name,
        )

    logger.info("Upserted %d points into collection %r", len(points), collection_name)


def build_all_qdrant_collections(
    passage_chunks: pd.DataFrame,
    sentence_chunks: pd.DataFrame,
    parent_child_chunks: pd.DataFrame,
) -> None:
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
        extra_payload_cols=("sentence_index", "prev_sent_id", "next_sent_id"),
    )
    upsert_collection(
        client,
        QDRANT_COLLECTIONS["child_chunks"],
        parent_child_chunks,
        extra_payload_cols=("parent_id", "parent_text"),
    )
