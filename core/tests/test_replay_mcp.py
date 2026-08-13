"""Replay substitutes MCP tool calls without ever invoking the real session.

The mirror of ``test_mcp.py``: where recording captures each ``call_tool``,
replaying answers it from the recording - and the session here would *raise* if
its real tool ran, so a green test is the proof that it never does.
"""

import asyncio
import contextlib
from typing import Any

from reenact.record import Recorder
from reenact.replay import Player
from reenact.replay.mcp import replaying
from reenact.schema import SideEffect


class _ExplodingSession:
    """An MCP session whose real tool call must never run during replay."""

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> Any:
        raise AssertionError("replay invoked the real MCP tool")


def _recorded_player() -> Player:
    rec = Recorder()
    rec.record_tool_call(
        name="search_docs",
        arguments={"q": "reenact"},
        result={"content": [{"type": "text", "text": "hit"}]},
        side_effect=SideEffect.READ_ONLY,
    )
    return Player(rec.trajectory)


def test_mcp_replay_substitutes_and_never_calls_the_real_tool() -> None:
    session = _ExplodingSession()

    async def run() -> Any:
        with replaying(session, _recorded_player()):
            result = await session.call_tool("search_docs", {"q": "reenact"})
            return result.model_dump()

    body = asyncio.run(run())
    assert body == {"content": [{"type": "text", "text": "hit"}]}


def test_mcp_replay_restores_call_tool_on_exit() -> None:
    session = _ExplodingSession()
    original = session.call_tool
    with replaying(session, _recorded_player()):
        pass
    assert session.call_tool == original
    assert "call_tool" not in vars(session)


def test_mcp_replay_restores_call_tool_on_exception() -> None:
    session = _ExplodingSession()
    original = session.call_tool

    with contextlib.suppress(RuntimeError), replaying(session, _recorded_player()):
        raise RuntimeError("boom")
    assert session.call_tool == original
