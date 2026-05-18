"""Disk-based SHA256 cache for LLM prompts and responses."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from config.settings import LLM_CACHE_DIR

logger = logging.getLogger(__name__)


def build_cache_key(*parts: str) -> str:
    """Create deterministic SHA256 cache key."""
    hasher = hashlib.sha256()

    for part in parts:
        hasher.update(part.encode("utf-8"))
        hasher.update(b"\x00")

    return hasher.hexdigest()


def cache_path(namespace: str, *parts: str) -> Path:
    """Return cache file path for a namespace and key."""
    digest = build_cache_key(*parts)

    cache_directory = (
        LLM_CACHE_DIR
        / namespace
        / digest[:2]
    )

    cache_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return cache_directory / f"{digest}.json"


def get_cached(namespace: str, *parts: str) -> Any | None:
    """Return cached object if present."""
    path = cache_path(namespace, *parts)

    if not path.exists():
        return None

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except (
        json.JSONDecodeError,
        OSError,
    ) as error:
        logger.warning(
            "Failed to read cache file %s: %s",
            path,
            error,
        )

        return None


def set_cached(
    namespace: str,
    value: Any,
    *parts: str,
) -> None:
    """Save object to cache."""
    path = cache_path(namespace, *parts)

    try:
        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                value,
                file,
                ensure_ascii=False,
                indent=2,
            )

    except OSError as error:
        logger.warning(
            "Failed to write cache file %s: %s",
            path,
            error,
        )
