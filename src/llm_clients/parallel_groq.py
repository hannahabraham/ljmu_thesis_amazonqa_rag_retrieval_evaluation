"""Parallel Groq client for batch prompt generation."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Result for one prompt."""

    answer: str | None
    latency_ms: float
    success: bool
    error: str | None = None


@dataclass
class SlidingWindowLimiter:
    """Sliding 60-second limiter for requests and tokens."""

    rpm: int
    tpm: int
    request_times: list[float] = field(default_factory=list)
    token_events: list[tuple[float, int]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def acquire(self, tokens: int) -> None:
        """Block until request/token budget is available."""
        while True:
            with self.lock:
                now = time.time()
                horizon = now - 60.0

                self.request_times = [
                    item for item in self.request_times if item > horizon
                ]

                self.token_events = [
                    item for item in self.token_events if item[0] > horizon
                ]

                request_count = len(self.request_times)
                token_count = sum(count for _, count in self.token_events)

                if request_count < self.rpm and token_count + tokens <= self.tpm:
                    self.request_times.append(now)
                    self.token_events.append((now, tokens))
                    return

                request_wait = (
                    self.request_times[0] + 60.0 - now
                    if request_count >= self.rpm
                    else 0.0
                )

                token_wait = (
                    self.token_events[0][0] + 60.0 - now
                    if token_count + tokens > self.tpm
                    else 0.0
                )

                wait_seconds = max(request_wait, token_wait, 0.05)

            time.sleep(wait_seconds)


def estimate_tokens(prompt: str, max_completion_tokens: int) -> int:
    """Estimate prompt + completion tokens."""
    return max(
        256,
        len(prompt) // 4 + max_completion_tokens,
    )


def is_fatal_error(error: BaseException) -> bool:
    """Return True if retrying the same key is unlikely to help."""
    message = str(error).lower()

    fatal_terms = (
        "invalid api key",
        "authentication",
        "401",
        "403",
        "permission denied",
        "billing",
    )

    return any(term in message for term in fatal_terms)


class ParallelGroqClient:
    """Dispatch prompts across multiple Groq API keys in parallel."""

    def __init__(
        self,
        api_keys: Sequence[str],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 200,
        rpm_per_key: int = 30,
        tpm_per_key: int = 12_000,
        max_retries: int = 4,
        request_timeout: int = 60,
    ) -> None:
        """Initialize one worker client per API key."""
        if not api_keys:
            raise ValueError("api_keys must be non-empty")

        from langchain_groq import ChatGroq

        self._model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._request_timeout = request_timeout

        self._workers: list[tuple[Any, SlidingWindowLimiter, str]] = []

        for api_key in api_keys:
            client = ChatGroq(
                groq_api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=request_timeout,
            )

            limiter = SlidingWindowLimiter(
                rpm=rpm_per_key,
                tpm=tpm_per_key,
            )

            self._workers.append(
                (
                    client,
                    limiter,
                    api_key[-6:],
                )
            )

    def call_one(self, client: Any, prompt: str) -> str:
        """Call one Groq client."""
        from langchain_core.messages import HumanMessage

        response = client.invoke(
            [HumanMessage(content=prompt)],
        )

        return str(response.content)

    def invoke_with_retry(
        self,
        client: Any,
        limiter: SlidingWindowLimiter,
        prompt: str,
        label: str,
        prompt_index: int,
    ) -> BatchResult:
        """Invoke one prompt with retry and rate limiting."""
        estimated_tokens = estimate_tokens(
            prompt=prompt,
            max_completion_tokens=self._max_tokens,
        )

        last_error: BaseException | None = None

        for attempt in range(self._max_retries):
            try:
                limiter.acquire(estimated_tokens)

                start_time = time.perf_counter()
                answer = self.call_one(client, prompt)
                latency_ms = (time.perf_counter() - start_time) * 1000.0

                return BatchResult(
                    answer=answer,
                    latency_ms=latency_ms,
                    success=True,
                )

            except Exception as error:  # noqa: BLE001
                last_error = error

                if is_fatal_error(error):
                    logger.error(
                        "Fatal Groq error. key=%s prompt=%d error=%s",
                        label,
                        prompt_index,
                        error,
                    )

                    return BatchResult(
                        answer=None,
                        latency_ms=0.0,
                        success=False,
                        error=str(error),
                    )

                wait_seconds = min(
                    30.0,
                    2.0**attempt,
                )

                logger.warning(
                    "Groq call failed. key=%s prompt=%d attempt=%d/%d "
                    "error=%s. Sleeping %.1fs.",
                    label,
                    prompt_index,
                    attempt + 1,
                    self._max_retries,
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
        """Dispatch prompts across workers and return structured results."""
        prompt_count = len(prompts)

        results = [
            BatchResult(
                answer=None,
                latency_ms=0.0,
                success=False,
                error="not processed",
            )
            for _ in range(prompt_count)
        ]

        work_queue: queue.Queue[tuple[int, str]] = queue.Queue()

        for index, prompt in enumerate(prompts):
            work_queue.put((index, prompt))

        def worker(
            client: Any,
            limiter: SlidingWindowLimiter,
            label: str,
        ) -> None:
            while True:
                try:
                    prompt_index, prompt = work_queue.get_nowait()
                except queue.Empty:
                    return

                try:
                    results[prompt_index] = self.invoke_with_retry(
                        client=client,
                        limiter=limiter,
                        prompt=prompt,
                        label=label,
                        prompt_index=prompt_index,
                    )
                finally:
                    work_queue.task_done()

        threads: list[threading.Thread] = []

        for client, limiter, label in self._workers:
            thread = threading.Thread(
                target=worker,
                args=(client, limiter, label),
                daemon=True,
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

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