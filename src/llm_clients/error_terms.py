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

DAILY_QUOTA_ERRORS: tuple[str, ...] = (
    "requests per day",
    "request per day",
    "per day",
    "daily",
    "rpd",
)


def is_daily_quota_error(error: BaseException) -> bool:
    """Return True when the error looks like a daily project quota stop."""
    message = str(error).lower()

    has_quota_context = any(
        term in message
        for term in (
            "quota",
            "resource exhausted",
            "resource_exhausted",
            "rate limit",
            "ratelimit",
            "429",
        )
    )

    return has_quota_context and any(
        term in message for term in DAILY_QUOTA_ERRORS
    )


def should_try_next_key(error: BaseException) -> bool:
    """Return True if retrying after backoff may succeed."""
    if is_daily_quota_error(error):
        return False

    message = str(error).lower()
    retry_terms = RETRY_BACKOFF_ERRORS + KEY_ROTATION_ERRORS

    return any(term in message for term in retry_terms)


def should_rotate_key(error: BaseException) -> bool:
    """Return True if trying another API key may succeed."""
    if is_daily_quota_error(error):
        return False

    message = str(error).lower()

    return any(term in message for term in KEY_ROTATION_ERRORS)
