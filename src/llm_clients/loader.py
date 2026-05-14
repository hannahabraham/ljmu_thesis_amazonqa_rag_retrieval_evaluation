"""Load one active API key from environment variables or the terminal."""

from __future__ import annotations

import getpass
import os
import re

from dotenv import load_dotenv

load_dotenv()


def prompt_for_key(service_name: str) -> str:
    """Ask the user for an API key without saving it to disk."""
    while True:
        value = getpass.getpass(f"Enter {service_name} API key: ").strip()
        if value:
            return value

        print("API key cannot be blank.")


def prompt_for_replacement_key(service_name: str, error: BaseException) -> str:
    """Ask for a replacement key after the active key fails."""
    print()
    print(f"{service_name} key failed or reached quota:")
    print(str(error))
    return prompt_for_key(f"new {service_name}")


def _numbered_key_sort(name: str) -> tuple[int, str]:
    match = re.search(r"_(\d+)$", name)
    if match:
        return int(match.group(1)), name

    return 0, name


def load_single_key(
    service_name: str,
    primary_env_name: str,
    numbered_prefix: str,
) -> str:
    """Load exactly one API key, preferring the unnumbered variable."""
    primary_value = os.getenv(primary_env_name, "").strip()
    if primary_value:
        return primary_value

    numbered_names = sorted(
        (
            name
            for name, value in os.environ.items()
            if name.startswith(numbered_prefix) and value.strip()
        ),
        key=_numbered_key_sort,
    )

    if numbered_names:
        return os.environ[numbered_names[0]].strip()

    return prompt_for_key(service_name)


def load_groq_key() -> str:
    """Load one Groq API key."""
    return load_single_key("Groq", "GROQ_API_KEY", "GROQ_API_KEY_")


def load_gemini_key() -> str:
    """Load one Gemini API key."""
    return load_single_key("Gemini", "GEMINI_API_KEY", "GEMINI_API_KEY_")


def load_groq_keys() -> list[str]:
    """Load one Groq API key, wrapped for existing call sites."""
    return [load_groq_key()]


def load_gemini_keys() -> list[str]:
    """Load one Gemini API key, wrapped for existing call sites."""
    return [load_gemini_key()]
