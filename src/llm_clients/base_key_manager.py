"""Multi-key LLM client with retry and rotation."""
from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Sequence

from src.llm_clients.error_terms import KEY_ROTATION_ERRORS, RETRY_BACKOFF_ERRORS

logger = logging.getLogger(__name__)


class BaseKeyManager(ABC):
    """Round-robin key rotation on quota errors, exponential jittered backoff on rate limits."""

    def __init__(
        self,
        api_keys: Sequence[str],
        model_name: str,
        temperature: float = 0.0,
        max_retries_per_key: int = 3,
        base_delay_seconds: float = 1.0,
        max_delay_seconds: float = 30.0,
    ) -> None:
        if not api_keys:
            raise ValueError("api_keys must be non-empty")
        self._api_keys = list(api_keys)
        self._model_name = model_name
        self._temperature = temperature
        self._max_retries_per_key = max_retries_per_key
        self._base_delay = base_delay_seconds
        self._max_delay = max_delay_seconds
        self._current_key_index = 0

    @property
    def current_key(self) -> str:
        return self._api_keys[self._current_key_index]

    def rotate_key(self) -> None:
        self._current_key_index = (self._current_key_index + 1) % len(self._api_keys)
        logger.info("Rotated to key index %d", self._current_key_index)

    @abstractmethod
    def _build_client(self, api_key: str) -> Any:
        """Build a fresh client for the given API key."""

    @abstractmethod
    def _call_client(self, client: Any, prompt: str) -> str:
        """Single completion call, returning the response text."""

    @staticmethod
    def classify_error(error: BaseException) -> str:
        message = str(error).lower()
        if any(term in message for term in KEY_ROTATION_ERRORS):
            return "rotate_key"
        if any(term in message for term in RETRY_BACKOFF_ERRORS):
            return "backoff"
        return "unknown"

    def _backoff_delay(self, retry_index: int) -> float:
        """Exponential backoff with full jitter (AWS pattern)."""
        capped = min(self._max_delay, self._base_delay * (2 ** retry_index))
        return random.uniform(0.0, capped)

    def invoke(self, prompt: str) -> str:
        last_error: BaseException | None = None
        for _ in range(len(self._api_keys)):
            client = self._build_client(self.current_key)
            rotated = False
            for retry in range(self._max_retries_per_key):
                try:
                    return self._call_client(client, prompt)
                except Exception as error:  # noqa: BLE001 -- provider-agnostic
                    last_error = error
                    kind = self.classify_error(error)
                    if kind == "backoff":
                        delay = self._backoff_delay(retry)
                        logger.warning(
                            "Backoff (%s): sleeping %.2fs (retry %d/%d)",
                            type(error).__name__, delay,
                            retry + 1, self._max_retries_per_key,
                        )
                        time.sleep(delay)
                        continue
                    if kind == "rotate_key":
                        logger.warning("Rotating key after error: %s", error)
                        self.rotate_key()
                        rotated = True
                        break
                    raise
            if not rotated:
                # Exhausted retries on this key without explicit rotation; rotate to give next key a chance.
                self.rotate_key()
        raise RuntimeError(f"All keys exhausted. Last error: {last_error}")
