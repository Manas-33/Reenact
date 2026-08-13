"""Shared machinery for provider capture adapters."""

import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

from reenact.record.recorder import Recorder
from reenact.schema import TokenUsage

type UsageExtractor = Callable[[dict[str, Any]], TokenUsage | None]


def capture_call(
    create: Any,
    recorder: Recorder,
    request: dict[str, Any],
    *,
    provider: str,
    usage_from: UsageExtractor,
) -> Any:
    """Call ``create(**request)``, record it, and return the response."""
    started = time.perf_counter()
    response = create(**request)
    latency_ms = (time.perf_counter() - started) * 1000.0
    body: dict[str, Any] = response.model_dump(mode="json")
    recorder.record_llm_call(
        provider=provider,
        model=str(request.get("model", "")),
        request=request,
        response=body,
        usage=usage_from(body),
        latency_ms=latency_ms,
    )
    return response


@contextmanager
def recording_on(
    owner: Any, *, provider: str, usage_from: UsageExtractor
) -> Generator[Recorder]:
    """Record every ``owner.create(**kwargs)`` call inside the block.

    ``owner`` holds the ``create`` method to wrap - an Anthropic
    ``client.messages`` or an OpenAI ``client.chat.completions``. The original
    method is restored on exit, even if the block raises.
    """
    recorder = Recorder()
    original = owner.create
    had_own = "create" in vars(owner)

    def _wrapped(**request: Any) -> Any:
        return capture_call(
            original, recorder, request, provider=provider, usage_from=usage_from
        )

    owner.create = _wrapped
    try:
        yield recorder
    finally:
        if had_own:
            owner.create = original
        else:
            del owner.create
