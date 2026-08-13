"""Replay tool calls made through an MCP client session - the mirror of
``record/mcp.py``.

Where recording swaps ``call_tool`` to capture each call, replaying swaps it to
answer each call from a recording. Every call is substituted from the trajectory,
so the real ``session.call_tool`` - and any side effect it would cause - is never
invoked. Async, because MCP is.
"""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from reenact.replay.player import Player


class _ReplayedResult:
    """A recorded tool result, dressed up so agent code can read it like a live
    MCP result (``result.model_dump()``) without an MCP dependency."""

    def __init__(self, body: Any) -> None:
        self._body = body

    def model_dump(self, *, mode: str = "python") -> Any:
        return self._body


@contextmanager
def replaying(session: Any, player: Player) -> Generator[Player]:
    """Substitute every ``session.call_tool(...)`` from ``player`` inside the block.

    The recorded result is handed back wrapped so ``result.model_dump()`` works;
    the real ``call_tool`` is never awaited, so replay causes no side effect. A
    call that no longer matches raises ``DivergenceError`` (strict) or is recorded
    on the player (lenient). ``call_tool`` is restored on exit, even on error.
    """
    original = session.call_tool
    had_own = "call_tool" in vars(session)

    async def _wrapped(
        name: str, arguments: dict[str, Any] | None = None, *args: Any, **kwargs: Any
    ) -> _ReplayedResult:
        return _ReplayedResult(player.replay_tool_call(name, arguments))

    session.call_tool = _wrapped
    try:
        yield player
    finally:
        if had_own:
            session.call_tool = original
        else:
            del session.call_tool
