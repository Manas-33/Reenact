"""Record calls made through an OpenAI client.

Duck-typed: works with the real ``openai.OpenAI`` client, or any object exposing
``chat.completions.create(**kwargs)`` that returns a response with a
``model_dump()`` method - so reenact does not depend on the SDK.
"""

from contextlib import AbstractContextManager
from typing import Any

from reenact.record._capture import capture_call, recording_on
from reenact.record.recorder import Recorder
from reenact.schema import TokenUsage


def _usage_from(response: dict[str, Any]) -> TokenUsage | None:
    """Pull OpenAI token usage (prompt/completion) out of a response body."""
    usage: Any = response.get("usage")
    if usage is None:
        return None
    return TokenUsage(
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
    )


def record_completion(client: Any, recorder: Recorder, **request: Any) -> Any:
    """Call ``client.chat.completions.create(**request)``, record it, return it.

    A drop-in around a single OpenAI chat completion: the returned object is the
    real response, unchanged, while ``recorder`` gains the captured event.
    """
    return capture_call(
        client.chat.completions.create,
        recorder,
        request,
        provider="openai",
        usage_from=_usage_from,
    )


def recording(client: Any) -> AbstractContextManager[Recorder]:
    """Record every OpenAI chat completion made through ``client`` in a block."""
    return recording_on(
        client.chat.completions, provider="openai", usage_from=_usage_from
    )
