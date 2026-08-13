"""Shared text helpers for the evals package: read answer text, clip a message."""

from typing import Any, cast

_MAX_MSG = 200


def clip(text: str, limit: int = _MAX_MSG) -> str:
    """Collapse whitespace and truncate long text for a one-line message."""
    flattened = " ".join(text.split())
    return flattened if len(flattened) <= limit else flattened[:limit] + "..."


def extract_text(response: dict[str, Any]) -> str:
    """Pull the assistant's text out of a recorded response body.

    Handles the two shapes the recorder stores: Anthropic content blocks
    (``content`` is a list, text lives in ``{"type": "text", "text": ...}``
    blocks) and OpenAI chat completions (``choices[0].message.content`` is a
    string). Non-text blocks (tool_use) are ignored.
    """
    content = response.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for block in cast(list[Any], content):
            if isinstance(block, dict):
                block_map = cast(dict[str, Any], block)
                text = block_map.get("text")
                if block_map.get("type") == "text" and isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    if isinstance(content, str):
        return content
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = cast(list[Any], choices)[0]
        if isinstance(first, dict):
            message = cast(dict[str, Any], first).get("message")
            if isinstance(message, dict):
                text = cast(dict[str, Any], message).get("content")
                if isinstance(text, str):
                    return text
    return ""
