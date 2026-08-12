"""Scrub sensitive values out of recorded request and response bodies."""

from typing import Any, cast

REDACTED = "[REDACTED]"

# Key names (matched case-insensitively) whose values are always scrubbed.
DEFAULT_SCRUB_KEYS: frozenset[str] = frozenset(
    {
        "authorization",
        "api_key",
        "api-key",
        "x-api-key",
        "anthropic-api-key",
        "openai-api-key",
        "cookie",
        "set-cookie",
        "password",
        "secret",
        "client_secret",
        "access_token",
        "refresh_token",
        "private_key",
    }
)


def redact(value: Any, scrub_keys: frozenset[str] = DEFAULT_SCRUB_KEYS) -> Any:
    """Return a copy of ``value`` with sensitive values replaced by ``REDACTED``.

    Recurses through dicts and lists. A dict entry whose key matches (case
    insensitively) an entry in ``scrub_keys`` has its value replaced; everything
    else is copied through unchanged. The input is never mutated.
    """
    if isinstance(value, dict):
        source = cast(dict[str, Any], value)
        return {
            key: REDACTED if key.lower() in scrub_keys else redact(item, scrub_keys)
            for key, item in source.items()
        }
    if isinstance(value, list):
        return [redact(item, scrub_keys) for item in cast(list[Any], value)]
    return value
