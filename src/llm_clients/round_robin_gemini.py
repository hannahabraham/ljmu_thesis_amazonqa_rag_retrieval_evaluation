"""Single-key Gemini chat model for RAGAS.

The exported class keeps the old name so existing imports keep working.
"""

from __future__ import annotations

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
from src.llm_clients.loader import prompt_for_replacement_key


class RoundRobinGeminiChat(BaseChatModel):
    """Gemini chat wrapper with one active key and interactive replacement."""

    keys: list[str]
    model: str
    temperature: float = 0.0
    max_retries: int = 1

    delegate: Any = Field(default=None, exclude=True)
    active_key: Any = Field(default=None, exclude=True)

    def __init__(self, **data: Any) -> None:
        """Initialize one Gemini chat client."""
        super().__init__(**data)

        api_key = str(next(iter(self.keys), "")).strip()
        if not api_key:
            raise ValueError("Gemini api key must be non-empty")

        object.__setattr__(self, "active_key", api_key)
        object.__setattr__(self, "delegate", self._build_delegate(api_key))

    @property
    def _llm_type(self) -> str:
        """Return custom LLM type name."""
        return "single_key_gemini"

    def _build_delegate(self, api_key: str) -> Any:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=self.model,
            google_api_key=api_key,
            temperature=self.temperature,
            max_retries=self.max_retries,
        )

    def _replace_key(self, error: BaseException) -> None:
        api_key = prompt_for_replacement_key("Gemini", error)
        object.__setattr__(self, "active_key", api_key)
        object.__setattr__(self, "delegate", self._build_delegate(api_key))

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response, asking for a new key on quota/auth errors."""
        last_error: BaseException | None = None

        for _ in range(self.max_retries + 1):
            try:
                return self.delegate._generate(
                    messages=messages,
                    stop=stop,
                    run_manager=run_manager,
                    **kwargs,
                )

            except BaseException as error:
                last_error = error
                if should_try_next_key(error):
                    self._replace_key(error)
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
        """Generate an async chat response with the active key."""
        last_error: BaseException | None = None

        for _ in range(self.max_retries + 1):
            try:
                return await self.delegate._agenerate(
                    messages=messages,
                    stop=stop,
                    run_manager=run_manager,
                    **kwargs,
                )

            except BaseException as error:
                last_error = error
                if should_try_next_key(error):
                    self._replace_key(error)
                    continue

                raise

        if last_error is not None:
            raise last_error

        raise RuntimeError("Async Gemini generation failed with no captured error.")
