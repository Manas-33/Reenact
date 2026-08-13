"""The MCP adapter records tool calls made through a client session."""

import asyncio
import contextlib
from typing import Any

from reenact.record import REDACTED, Recorder
from reenact.record.mcp import record_call_tool, recording
from reenact.schema import SideEffect, ToolCallEvent


class _Result:
    """Stand-in for an MCP CallToolResult (a pydantic-style model)."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        return self._payload


class _StubSession:
    """A duck-typed MCP session: async ``call_tool`` returning a result."""

    def __init__(self, result: Any) -> None:
        self._result = result
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> Any:
        self.calls.append((name, arguments))
        return self._result


def test_record_call_tool_records_and_returns_the_real_result() -> None:
    result = _Result({"content": [{"type": "text", "text": "42"}], "isError": False})
    session = _StubSession(result)
    rec = Recorder()

    returned = asyncio.run(
        record_call_tool(
            session, rec, "compute", {"a": 1}, side_effect=SideEffect.READ_ONLY
        )
    )

    assert returned is result  # drop-in: real result, unchanged
    event = rec.trajectory.events[0]
    assert isinstance(event, ToolCallEvent)
    assert event.name == "compute"
    assert event.arguments == {"a": 1}
    assert event.result == {
        "content": [{"type": "text", "text": "42"}],
        "isError": False,
    }
    assert event.side_effect is SideEffect.READ_ONLY


def test_side_effect_defaults_to_unknown() -> None:
    session = _StubSession(_Result({"content": []}))
    rec = Recorder()
    asyncio.run(record_call_tool(session, rec, "do_thing"))
    event = rec.trajectory.events[0]
    assert isinstance(event, ToolCallEvent)
    assert event.side_effect is SideEffect.UNKNOWN
    assert event.arguments == {}


def test_arguments_are_redacted() -> None:
    session = _StubSession(_Result({"content": []}))
    rec = Recorder()
    asyncio.run(record_call_tool(session, rec, "call_api", {"api_key": "sk-secret"}))
    event = rec.trajectory.events[0]
    assert isinstance(event, ToolCallEvent)
    assert event.arguments["api_key"] == REDACTED


def test_recording_wraps_every_call_and_restores_call_tool() -> None:
    session = _StubSession(_Result({"content": [{"type": "text", "text": "ok"}]}))
    original = session.call_tool

    async def run() -> Recorder:
        with recording(session) as rec:
            await session.call_tool("a", {"x": 1})
            await session.call_tool("b", None)
        return rec

    rec = asyncio.run(run())

    names = [e.name for e in rec.trajectory.events if isinstance(e, ToolCallEvent)]
    assert names == ["a", "b"]
    assert session.call_tool == original  # restored on exit
    assert "call_tool" not in vars(session)
    assert session.calls == [("a", {"x": 1}), ("b", None)]


def test_recording_restores_call_tool_on_exception() -> None:
    session = _StubSession(_Result({"content": []}))
    original = session.call_tool

    async def run() -> None:
        with recording(session):
            await session.call_tool("a")
            raise RuntimeError("boom")

    with contextlib.suppress(RuntimeError):
        asyncio.run(run())
    assert session.call_tool == original
    assert "call_tool" not in vars(session)
