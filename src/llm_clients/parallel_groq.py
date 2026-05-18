"""Single-key Groq client for batch prompt generation.

Prompts are processed sequentially with one active API key. When a quota or
auth error is raised, the user is prompted (via ``getpass``) for a
replacement key. Multi-key rotation was removed -- the thesis pipeline is
re-runnable with a fresh key whenever the current one is exhausted.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Sequence

from src.llm_clients.error_terms import should_try_next_key
from src.llm_clients.loader import prompt_for_replacement_key

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Result for one prompt."""

    answer: str | None
    latency_ms: float
    success: bool
    error: str | None = None


class GroqClient:
    """Generate with one Groq key at a time, asking for a replacement on quota."""

    def __init__(
        self,
        api_keys: Sequence[str] | str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 200,
        max_retries: int = 2,
        request_timeout: int = 60,
    ) -> None:
        """Initialize the active Groq client.

        ``api_keys`` accepts either a single string or a sequence (only the
        first non-empty entry is used). The sequence form is preserved for
        backwards compatibility with call sites that still pass a list.
        """
        if isinstance(api_keys, str):
            api_key = api_keys.strip()
        else:
            api_key = str(next(iter(api_keys), "")).strip()
        if not api_key:
            raise ValueError("Groq api key must be non-empty")

        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._request_timeout = request_timeout
        self._max_retries = max_retries
        self._api_key = api_key
        self._client = self._build_client(api_key)

    def _build_client(self, api_key: str) -> Any:
        from langchain_groq import ChatGroq

        return ChatGroq(
            groq_api_key=api_key,
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            timeout=self._request_timeout,
            max_retries=0,
        )

    def _replace_key(self, error: BaseException) -> None:
        self._api_key = prompt_for_replacement_key("Groq", error)
        self._client = self._build_client(self._api_key)
        logger.info("Groq API key replaced by user input.")

    def call_one(self, prompt: str) -> str:
        """Call Groq once."""
        from langchain_core.messages import HumanMessage

        response = self._client.invoke(
            [HumanMessage(content=prompt)],
        )

        return str(response.content)

    def invoke_with_retry(self, prompt: str, prompt_index: int) -> BatchResult:
        """Invoke one prompt, asking for a new key on quota/auth failures."""
        last_error: BaseException | None = None

        for attempt in range(self._max_retries + 1):
            try:
                start_time = time.perf_counter()
                answer = self.call_one(prompt)
                latency_ms = (time.perf_counter() - start_time) * 1000.0

                return BatchResult(
                    answer=answer,
                    latency_ms=latency_ms,
                    success=True,
                )

            except Exception as error:  # noqa: BLE001 -- provider SDK errors vary.
                last_error = error

                if should_try_next_key(error):
                    logger.warning(
                        "Groq key failed or reached quota at prompt %d: %s",
                        prompt_index,
                        error,
                    )
                    self._replace_key(error)
                    continue

                wait_seconds = min(15.0, 2.0**attempt)
                logger.warning(
                    "Groq call failed at prompt %d attempt %d/%d: %s. Sleeping %.1fs.",
                    prompt_index,
                    attempt + 1,
                    self._max_retries + 1,
                    type(error).__name__,
                    wait_seconds,
                )
                time.sleep(wait_seconds)

        return BatchResult(
            answer=None,
            latency_ms=0.0,
            success=False,
            error=str(last_error),
        )

    def batch_invoke(
        self,
        prompts: Sequence[str],
    ) -> tuple[list[str | None], list[float]]:
        """Return answers and latencies in input order."""
        batch_results = self.batch_invoke_structured(prompts)

        answers = [
            result.answer if result.success else None
            for result in batch_results
        ]

        latencies = [
            result.latency_ms
            for result in batch_results
        ]

        return answers, latencies

    def batch_invoke_structured(
        self,
        prompts: Sequence[str],
    ) -> list[BatchResult]:
        """Process prompts sequentially with the active key."""
        results: list[BatchResult] = []
        prompt_count = len(prompts)

        for index, prompt in enumerate(prompts):
            logger.info("Calling Groq prompt %d/%d", index + 1, prompt_count)
            results.append(
                self.invoke_with_retry(
                    prompt=prompt,
                    prompt_index=index,
                )
            )

        failed_count = sum(
            not result.success
            for result in results
        )

        if failed_count:
            logger.warning(
                "%d/%d Groq prompts failed.",
                failed_count,
                prompt_count,
            )

        return results


# Backwards-compatible alias for the legacy import path.
ParallelGroqClient = GroqClient

