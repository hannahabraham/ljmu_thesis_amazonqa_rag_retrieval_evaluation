"""Round-robin Gemini chat model for RAGAS.

The wrapper holds every configured Gemini key and rotates to the next one
whenever the active key returns a retryable error (quota, rate limit,
timeout, or transient 5xx). After every key has been tried in a single
call it either prompts for a replacement (interactive mode) or raises so
the caller's outer backoff can absorb the wait.

The legacy ``RoundRobinGeminiChat`` / ``SingleKeyGeminiChat`` import names
are kept as aliases for backwards compatibility.
"""

from __future__ import annotations

import logging
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

LOGGER = logging.getLogger(__name__)


class RotatingGeminiChat(BaseChatModel):
    """Gemini chat wrapper that round-robins through multiple keys."""

    keys: list[str]
    model: str
    temperature: float = 0.0
    max_retries: int = 1
    interactive_replacement: bool = True

    delegate: Any = Field(default=None, exclude=True)
    active_key: Any = Field(default=None, exclude=True)
    active_index: int = Field(default=0, exclude=True)

    def __init__(self, **data: Any) -> None:
        """Initialize the wrapper with the first key as the active delegate."""
        super().__init__(**data)

        cleaned = [str(key).strip() for key in self.keys if str(key).strip()]
        if not cleaned:
            raise ValueError("Gemini api key list must be non-empty")

        object.__setattr__(self, "keys", cleaned)
        object.__setattr__(self, "active_index", 0)
        object.__setattr__(self, "active_key", cleaned[0])
        object.__setattr__(self, "delegate", self._build_delegate(cleaned[0]))

        LOGGER.info(
            "RotatingGeminiChat initialised with %d Gemini key(s)",
            len(cleaned),
        )

    @property
    def _llm_type(self) -> str:
        """Return the custom LLM type name."""
        return "rotating_gemini"

    def _build_delegate(self, api_key: str) -> Any:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=self.model,
            google_api_key=api_key,
            temperature=self.temperature,
            max_retries=self.max_retries,
        )

    def _rotate_to_next_key(self, error: BaseException) -> None:
        """Advance to the next key in the rotation and rebuild the delegate."""
        next_index = (self.active_index + 1) % len(self.keys)
        next_key = self.keys[next_index]
        LOGGER.warning(
            "Rotating Gemini key %d -> %d after retryable error: %s",
            self.active_index + 1,
            next_index + 1,
            error,
        )
        object.__setattr__(self, "active_index", next_index)
        object.__setattr__(self, "active_key", next_key)
        object.__setattr__(self, "delegate", self._build_delegate(next_key))

    def _replace_key(self, error: BaseException) -> None:
        """Prompt the user for a new key and swap it into the active slot."""
        api_key = prompt_for_replacement_key("Gemini", error)
        keys = list(self.keys)
        keys[self.active_index] = api_key
        object.__setattr__(self, "keys", keys)
        object.__setattr__(self, "active_key", api_key)
        object.__setattr__(self, "delegate", self._build_delegate(api_key))

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response, rotating keys on retryable errors."""
        last_error: BaseException | None = None

        for _ in range(len(self.keys)):
            try:
                # Delegate to ChatGoogleGenerativeAI._generate directly: RAGAS
                # consumes the ChatResult return type from this private method,
                # and the public .invoke() wraps it in a higher-level object
                # that RAGAS doesn't accept. If a future langchain release
                # renames `_generate`, this call will need to update too.
                return self.delegate._generate(
                    messages=messages,
                    stop=stop,
                    run_manager=run_manager,
                    **kwargs,
                )
            except Exception as error:  # noqa: BLE001 -- langchain SDK errors vary
                last_error = error
                if not should_try_next_key(error):
                    raise
                self._rotate_to_next_key(error)

        if self.interactive_replacement and last_error is not None:
            self._replace_key(last_error)
            return self.delegate._generate(
                messages=messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )

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
        """Generate an async chat response, rotating keys on retryable errors."""
        last_error: BaseException | None = None

        for _ in range(len(self.keys)):
            try:
                return await self.delegate._agenerate(
                    messages=messages,
                    stop=stop,
                    run_manager=run_manager,
                    **kwargs,
                )
            except Exception as error:  # noqa: BLE001 -- langchain SDK errors vary
                last_error = error
                if not should_try_next_key(error):
                    raise
                self._rotate_to_next_key(error)

        if self.interactive_replacement and last_error is not None:
            self._replace_key(last_error)
            return await self.delegate._agenerate(
                messages=messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )

        if last_error is not None:
            raise last_error

        raise RuntimeError("Async Gemini generation failed with no captured error.")


# Backwards-compatible aliases for the legacy import paths.
SingleKeyGeminiChat = RotatingGeminiChat
RoundRobinGeminiChat = RotatingGeminiChat
