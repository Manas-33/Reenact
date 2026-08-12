"""Record calls made through an Anthropic client.

Duck-typed: works with the real ``anthropic.Anthropic`` client, or any object
exposing ``messages.create(**kwargs)`` that returns a response with a
``model_dump()`` method - so reenact does not depend on the SDK.
"""

import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from reenact.record.recorder import Recorder
from reenact.schema import TokenUsage


def _usage_from(response: dict[str, Any]) -> TokenUsage | None:
    """Pull Anthropic token usage out of a response body, if present."""
    usage: Any = response.get("usage")
    if usage is None:
        return None
    return TokenUsage(
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
    )


def _record_call(create: Any, recorder: Recorder, request: dict[str, Any]) -> Any:
    """Call ``create(**request)``, record it, and return the response."""
    started = time.perf_counter()
    response = create(**request)
    latency_ms = (time.perf_counter() - started) * 1000.0
    body: dict[str, Any] = response.model_dump(mode="json")
    recorder.record_llm_call(
        provider="anthropic",
        model=str(request.get("model", "")),
        request=request,
        response=body,
        usage=_usage_from(body),
        latency_ms=latency_ms,
    )
    return response


def record_message(client: Any, recorder: Recorder, **request: Any) -> Any:
    """Call ``client.messages.create(**request)``, record it, return the response.

    A drop-in around a single Anthropic call: the returned object is the real
    response, unchanged, while ``recorder`` gains the captured event.
    """
    return _record_call(client.messages.create, recorder, request)


@contextmanager
def recording(client: Any) -> Generator[Recorder]:
    """Record every Anthropic call made through ``client`` inside the block.

    Temporarily wraps ``client.messages.create`` so agent code inside the
    ``with`` block needs no changes; the original method is restored on exit,
    even if the block raises.
    """
    recorder = Recorder()
    messages = client.messages
    original = messages.create
    had_own = "create" in vars(messages)

    def _wrapped(**request: Any) -> Any:
        return _record_call(original, recorder, request)

    messages.create = _wrapped
    try:
        yield recorder
    finally:
        if had_own:
            messages.create = original
        else:
            del messages.create
