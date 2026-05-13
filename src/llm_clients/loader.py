"""Load API keys from environment variables."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def load_keys_with_prefix(prefix: str) -> list[str]:
    """Load all environment variables matching a prefix."""
    keys: list[str] = []

    for env_key, env_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue

        cleaned_value = env_value.strip()

        if cleaned_value:
            keys.append(cleaned_value)

    if not keys:
        raise ValueError(
            f"No environment variables found with prefix: {prefix}",
        )

    return sorted(keys)


def load_groq_keys() -> list[str]:
    """Load Groq API keys."""
    return load_keys_with_prefix("GROQ_API_KEY")


def load_gemini_keys() -> list[str]:
    """Load Gemini API keys."""
    return load_keys_with_prefix("GEMINI_API_KEY")