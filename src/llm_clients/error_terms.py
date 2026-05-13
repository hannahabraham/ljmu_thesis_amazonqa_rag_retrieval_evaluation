"""Shared error-string heuristics for LLM key managers."""

from __future__ import annotations

RETRY_BACKOFF_ERRORS: tuple[str, ...] = (
    "ratelimit",
    "rate limit",
    "429",
    "too many requests",
    "capacity",
    "overloaded",
    "503",
    "502",
    "504",
    "timeout",
    "deadline exceeded",
)

KEY_ROTATION_ERRORS: tuple[str, ...] = (
    "quota",
    "resource exhausted",
    "resource_exhausted",
    "billing",
    "exceeded",
    "permission denied",
    "401",
    "403",
    "invalid api key",
    "authentication",
)


def should_try_next_key(error: BaseException) -> bool:
    """Return True if retrying with another key may succeed."""
    message = str(error).lower()
    retry_terms = RETRY_BACKOFF_ERRORS + KEY_ROTATION_ERRORS

    return any(term in message for term in retry_terms)