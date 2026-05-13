"""LangChain chat model that round-robins Gemini API keys across calls."""

from __future__ import annotations

import threading
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from pydantic import Field

from src.llm_clients.error_terms import should_try_next_key


class RoundRobinGeminiChat(BaseChatModel):
    """Chat model wrapper that rotates Gemini API keys per call."""

    keys: list[str]
    model: str
    temperature: float = 0.0
    max_retries: int = 1

    delegates: Any = Field(default=None, exclude=True)
    key_index: Any = Field(default=None, exclude=True)
    lock: Any = Field(default=None, exclude=True)

    def __init__(self, **data: Any) -> None:
        """Initialize one Gemini chat client per API key."""
        super().__init__(**data)

        if not self.keys:
            raise ValueError("keys must be non-empty")

        from langchain_google_genai import ChatGoogleGenerativeAI

        delegates = [
            ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=key,
                temperature=self.temperature,
                max_retries=self.max_retries,
            )
            for key in self.keys
        ]

        object.__setattr__(self, "delegates", delegates)
        object.__setattr__(self, "key_index", [0])
        object.__setattr__(self, "lock", threading.Lock())

    @property
    def _llm_type(self) -> str:
        """Return custom LLM type name."""
        return "round_robin_gemini"

    def next_llm(self) -> Any:
        """Return the next Gemini delegate."""
        with self.lock:
            index = self.key_index[0]
            self.key_index[0] = (index + 1) % len(self.delegates)

        return self.delegates[index]

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response with key rotation."""
        last_error: BaseException | None = None

        for _ in range(len(self.delegates)):
            llm = self.next_llm()

            try:
                return llm._generate(
                    messages=messages,
                    stop=stop,
                    run_manager=run_manager,
                    **kwargs,
                )

            except BaseException as error:
                if should_try_next_key(error):
                    last_error = error
                    continue

                raise

        if last_error is not None:
            raise last_error

        raise RuntimeError("Gemini generation failed with no captured error.")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate an async chat response with key rotation."""
        last_error: BaseException | None = None

        for _ in range(len(self.delegates)):
            llm = self.next_llm()

            try:
                return await llm._agenerate(
                    messages=messages,
                    stop=stop,
                    run_manager=run_manager,
                    **kwargs,
                )

            except BaseException as error:
                if should_try_next_key(error):
                    last_error = error
                    continue

                raise

        if last_error is not None:
            raise last_error

        raise RuntimeError("Async Gemini generation failed with no captured error.")