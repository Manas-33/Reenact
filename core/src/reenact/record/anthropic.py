"""Record calls made through an Anthropic client.

Duck-typed: works with the real ``anthropic.Anthropic`` client, or any object
exposing ``messages.create(**kwargs)`` that returns a response with a
``model_dump()`` method - so reenact does not depend on the SDK.
"""

import time
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


def record_message(client: Any, recorder: Recorder, **request: Any) -> Any:
    """Call ``client.messages.create(**request)``, record it, return the response.

    A drop-in around the Anthropic call: the returned object is the real
    response, unchanged, while ``recorder`` gains the captured event.
    """
    started = time.perf_counter()
    response = client.messages.create(**request)
    latency_ms = (time.perf_counter() - started) * 1000.0
    body: dict[str, Any] = response.model_dump(mode="json")
    recorder.record_llm_call(
        provider="anthropic",
        model=str(request.get("model", "")),
        request=dict(request),
        response=body,
        usage=_usage_from(body),
        latency_ms=latency_ms,
    )
    return response
