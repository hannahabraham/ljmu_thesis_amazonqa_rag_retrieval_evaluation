"""Single-key Gemini client using the google-genai SDK."""

from __future__ import annotations

import logging
from typing import Any, Sequence

from src.llm_clients.error_terms import should_try_next_key
from src.llm_clients.loader import prompt_for_replacement_key

LOGGER = logging.getLogger(__name__)


class GeminiKeyManager:
    """Call Gemini with one active key, prompting when that key cannot continue."""

    def __init__(
        self,
        api_keys: str | Sequence[str],
        model_name: str,
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> None:
        """Initialise the manager with one or more API keys (first non-empty wins)."""
        if isinstance(api_keys, str):
            api_key = api_keys.strip()
        else:
            api_key = str(next(iter(api_keys), "")).strip()

        if not api_key:
            raise ValueError("Gemini api key must be non-empty")

        self._api_key = api_key
        self._model_name = model_name
        self._temperature = temperature
        self._max_retries = max_retries
        self._client = self._build_client(self._api_key)

    def _build_client(self, api_key: str) -> Any:
        """Create a Gemini client for one API key."""
        from google import genai

        return genai.Client(api_key=api_key)

    def _call_client(self, prompt: str) -> str:
        """Call Gemini and return response text."""
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config={
                "temperature": self._temperature,
            },
        )

        if response.text is None:
            return ""

        return response.text

    def _replace_key(self, error: BaseException) -> None:
        self._api_key = prompt_for_replacement_key("Gemini", error)
        self._client = self._build_client(self._api_key)
        LOGGER.info("Gemini API key replaced by user input.")

    def invoke(self, prompt: str) -> str:
        """Generate content, asking for a new key on quota/auth failures."""
        last_error: BaseException | None = None

        for _ in range(self._max_retries + 1):
            try:
                return self._call_client(prompt)
            except Exception as error:  # noqa: BLE001 -- provider SDK errors vary.
                last_error = error
                if should_try_next_key(error):
                    self._replace_key(error)
                    continue

                raise

        raise RuntimeError(f"Gemini call failed after key replacement: {last_error}")
