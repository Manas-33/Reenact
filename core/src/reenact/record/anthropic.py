"""Record calls made through an Anthropic client.

Duck-typed: works with the real ``anthropic.Anthropic`` client, or any object
exposing ``messages.create(**kwargs)`` that returns a response with a
``model_dump()`` method - so reenact does not depend on the SDK.
"""

from contextlib import AbstractContextManager
from typing import Any

from reenact.record._capture import capture_call, recording_on
from reenact.record.recorder import Recorder
from reenact.schema import TokenUsage


def _usage_from(response: dict[str, Any]) -> TokenUsage | None:
    """Pull Anthropic token usage (input/output) out of a response body."""
    usage: Any = response.get("usage")
    if usage is None:
        return None
    return TokenUsage(
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
    )


def record_message(client: Any, recorder: Recorder, **request: Any) -> Any:
    """Call ``client.messages.create(**request)``, record it, return the response.

    A drop-in around a single Anthropic call: the returned object is the real
    response, unchanged, while ``recorder`` gains the captured event.
    """
    return capture_call(
        client.messages.create,
        recorder,
        request,
        provider="anthropic",
        usage_from=_usage_from,
    )


def recording(client: Any) -> AbstractContextManager[Recorder]:
    """Record every Anthropic call made through ``client`` in a ``with`` block."""
    return recording_on(client.messages, provider="anthropic", usage_from=_usage_from)
