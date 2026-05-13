"""Behavioural tests for ``BaseKeyManager`` using a fake subclass."""

from __future__ import annotations

from typing import Any

import pytest

from src.llm_clients.base_key_manager import BaseKeyManager
from src.llm_clients.error_terms import KEY_ROTATION_ERRORS, RETRY_BACKOFF_ERRORS


class FakeManager(BaseKeyManager):
    """Fake concrete key manager for offline behavioural tests."""

    def __init__(
        self,
        keys: list[str],
        *,
        behaviours: list[Any],
        **kwargs: Any,
    ) -> None:
        """Initialise the fake manager."""
        super().__init__(
            keys,
            model_name="fake-model",
            base_delay_seconds=0.0,
            **kwargs,
        )
        self._behaviours = behaviours
        self._calls = 0

    def _build_client(self, api_key: str) -> dict[str, str]:
        """Build a fake client for the supplied API key."""
        return {"key": api_key}

    def _call_client(self, client: dict[str, str], prompt: str) -> str:
        """Return configured fake responses or raise configured exceptions."""
        del prompt

        call_index = self._calls
        self._calls += 1

        if call_index >= len(self._behaviours):
            return f"ok:{self._calls}:{client['key']}"

        behaviour = self._behaviours[call_index]

        if isinstance(behaviour, Exception):
            raise behaviour

        return behaviour


def test_success_first_call() -> None:
    """Test successful response on the first call."""
    manager = FakeManager(["k1", "k2"], behaviours=["yay"])

    assert manager.invoke("hi") == "yay"


def test_backoff_on_rate_limit_then_succeed() -> None:
    """Test retry backoff on rate-limit-like errors before success."""
    behaviours = [Exception("rate limit hit"), Exception("429"), "finally"]
    manager = FakeManager(["k1"], behaviours=behaviours)

    assert manager.invoke("hi") == "finally"


def test_rotation_on_quota_error() -> None:
    """Test API key rotation when a quota-like error is raised."""
    behaviours = [Exception("quota exceeded"), "ok-from-second-key"]
    manager = FakeManager(["k1", "k2"], behaviours=behaviours)

    assert manager.invoke("hi") == "ok-from-second-key"
    assert manager.current_key == "k2"


def test_unknown_error_propagates() -> None:
    """Test unknown errors are propagated without retry or rotation."""
    behaviours = [ValueError("definitely not a rate limit")]
    manager = FakeManager(["k1", "k2"], behaviours=behaviours)

    with pytest.raises(ValueError):
        manager.invoke("hi")


def test_all_keys_exhausted() -> None:
    """Test RuntimeError is raised when all available keys are exhausted."""
    behaviours = [Exception("quota exceeded")] * 6
    manager = FakeManager(
        ["k1", "k2"],
        behaviours=behaviours,
        max_retries_per_key=1,
    )

    with pytest.raises(RuntimeError, match="All keys exhausted"):
        manager.invoke("hi")


def test_empty_keys_raises() -> None:
    """Test manager initialisation fails when no keys are provided."""
    with pytest.raises(ValueError):
        FakeManager([], behaviours=[])


def test_classify_error_recognises_each_term() -> None:
    """Test configured retry and rotation error terms are recognised."""
    for term in KEY_ROTATION_ERRORS:
        error_type = BaseKeyManager.classify_error(Exception(f"saw {term} here"))
        assert error_type == "rotate_key"

    for term in RETRY_BACKOFF_ERRORS:
        error_type = BaseKeyManager.classify_error(Exception(f"saw {term} here"))
        assert error_type == "backoff"


def test_backoff_delay_within_bounds() -> None:
    """Test calculated backoff delays stay within configured bounds."""
    manager = FakeManager(
        ["k1"],
        behaviours=["x"],
        base_delay_seconds=0.5,
        max_delay_seconds=2.0,
    )

    for retry in range(5):
        delay = manager._backoff_delay(retry)  # pylint: disable=protected-access
        assert 0.0 <= delay <= 2.0