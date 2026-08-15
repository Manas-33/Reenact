"""Shared text helpers for the evals package: read answer text, clip a message."""

from typing import Any, cast

_MAX_MSG = 200


def clip(text: str, limit: int = _MAX_MSG) -> str:
    """Collapse whitespace and truncate long text for a one-line message."""
    flattened = " ".join(text.split())
    return flattened if len(flattened) <= limit else flattened[:limit] + "..."


def extract_text(response: dict[str, Any]) -> str:
    """Pull the assistant's text out of a recorded response body.

    Handles the three shapes the recorders store: Anthropic content blocks
    (``content`` is a list, text lives in ``{"type": "text", "text": ...}``
    blocks), OpenAI chat completions (``choices[0].message.content`` is a string),
    and LangChain's ``LLMResult`` (``generations[i][j].text``, what the LangGraph
    callback handler records). Non-text blocks (tool_use) are ignored.
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
    generations = response.get("generations")
    if isinstance(generations, list):
        texts: list[str] = []
        for batch in cast(list[Any], generations):
            if isinstance(batch, list):
                for generation in cast(list[Any], batch):
                    if isinstance(generation, dict):
                        text = cast(dict[str, Any], generation).get("text")
                        if isinstance(text, str) and text:
                            texts.append(text)
        return "".join(texts)
    return ""


def response_text(response: Any) -> str:
    """Pull the reply text from a judge/evaluator client's response.

    The client is duck-typed (the same rule as the SDK recorders), so the reply is
    either an SDK object exposing ``model_dump`` or a plain dict. Either way the
    text is read through :func:`extract_text`. Raises ``TypeError`` on a shape that
    is neither - a broken client should fail loudly, not score silently.
    """
    if hasattr(response, "model_dump"):
        body: Any = response.model_dump(mode="json")
        if isinstance(body, dict):
            return extract_text(cast(dict[str, Any], body))
    if isinstance(response, dict):
        return extract_text(cast(dict[str, Any], response))
    raise TypeError("client returned an unreadable response")
