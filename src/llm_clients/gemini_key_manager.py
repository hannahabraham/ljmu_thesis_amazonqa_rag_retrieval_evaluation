"""Gemini key manager using the google-genai SDK."""

from __future__ import annotations

from typing import Any

from src.llm_clients.base_key_manager import BaseKeyManager


class GeminiKeyManager(BaseKeyManager):
    """Manage Gemini API keys and model calls."""

    def _build_client(self, api_key: str) -> Any:
        """Create a Gemini client for one API key."""
        from google import genai

        return genai.Client(api_key=api_key)

    def _call_client(self, client: Any, prompt: str) -> str:
        """Call Gemini and return response text."""
        response = client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config={
                "temperature": self._temperature,
            },
        )

        if response.text is None:
            return ""

        return response.text