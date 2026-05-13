"""Sanity-check Gemini manager wiring without importing google-genai.

The abstract methods are patched to keep tests offline. Behavioural retry
coverage lives in ``test_base_key_manager.py``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


def test_gemini_invoke_via_patch() -> None:
    """Test GeminiKeyManager invokes successfully with patched client methods."""
    from src.llm_clients.gemini_key_manager import GeminiKeyManager

    with (
        patch.object(GeminiKeyManager, "_build_client", return_value={"k": "x"}),
        patch.object(GeminiKeyManager, "_call_client", return_value="ok"),
    ):
        manager = GeminiKeyManager(["k1"], "gemini-2.5-flash")

        assert manager.invoke("hi") == "ok"


def test_gemini_requires_keys() -> None:
    """Test GeminiKeyManager raises when no API keys are provided."""
    from src.llm_clients.gemini_key_manager import GeminiKeyManager

    with pytest.raises(ValueError):
        GeminiKeyManager([], "gemini-2.5-flash")