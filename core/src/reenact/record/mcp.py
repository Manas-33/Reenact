"""Record tool calls made through an MCP client session.

Duck-typed: works with an MCP ``ClientSession`` - or any object exposing an
async ``call_tool(name, arguments)`` that returns a result with a
``model_dump()`` method - so reenact does not depend on the MCP SDK.

MCP tool calls are async, so these adapters are ``async``; otherwise they mirror
the SDK adapters: a drop-in that times the call, records a ``ToolCallEvent``,
and returns the real result unchanged.
"""

import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from reenact.record.recorder import Recorder
from reenact.schema import SideEffect


def _result_body(result: Any) -> Any:
    """Serialize an MCP tool result for storage; pass through anything plain."""
    return result.model_dump(mode="json") if hasattr(result, "model_dump") else result


async def record_call_tool(
    session: Any,
    recorder: Recorder,
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    side_effect: SideEffect = SideEffect.UNKNOWN,
) -> Any:
    """Call ``session.call_tool(name, arguments)``, record it, return the result.

    A drop-in around a single MCP tool call: the returned object is the real
    result, unchanged, while ``recorder`` gains the captured event. ``side_effect``
    defaults to ``UNKNOWN``, which replay treats as mutating (the safe default).
    """
    started = time.perf_counter()
    result = await session.call_tool(name, arguments)
    latency_ms = (time.perf_counter() - started) * 1000.0
    recorder.record_tool_call(
        name=name,
        arguments=arguments if arguments is not None else {},
        result=_result_body(result),
        side_effect=side_effect,
        latency_ms=latency_ms,
    )
    return result


@contextmanager
def recording(
    session: Any, *, side_effect: SideEffect = SideEffect.UNKNOWN
) -> Generator[Recorder]:
    """Record every ``session.call_tool(...)`` made inside the block.

    Monkeypatches the session's ``call_tool`` and restores it on exit, even if
    the block raises. A blanket wrapper cannot know each tool's effect, so every
    call is recorded with ``side_effect`` (``UNKNOWN`` by default, i.e. mutating).
    """
    recorder = Recorder()
    original = session.call_tool
    had_own = "call_tool" in vars(session)

    async def _wrapped(
        name: str, arguments: dict[str, Any] | None = None, *args: Any, **kwargs: Any
    ) -> Any:
        started = time.perf_counter()
        result = await original(name, arguments, *args, **kwargs)
        latency_ms = (time.perf_counter() - started) * 1000.0
        recorder.record_tool_call(
            name=name,
            arguments=arguments if arguments is not None else {},
            result=_result_body(result),
            side_effect=side_effect,
            latency_ms=latency_ms,
        )
        return result

    session.call_tool = _wrapped
    try:
        yield recorder
    finally:
        if had_own:
            session.call_tool = original
        else:
            del session.call_tool
