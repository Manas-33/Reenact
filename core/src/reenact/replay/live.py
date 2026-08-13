"""Route a client's LLM calls through a recorded trajectory instead of the network.

The mirror image of ``reenact.recording``: where recording swaps a client's
``create`` to capture calls, ``replaying`` swaps it to answer them from a
recording. Agent code runs unchanged, fully offline; a call that no longer
matches the recording raises ``DivergenceError``.
"""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from reenact.replay.player import Player, ReplayMode
from reenact.schema import Trajectory


class _ReplayedResponse:
    """A recorded response body, dressed up so agent code can read it like a
    live SDK response (``response.model_dump()``) without an SDK dependency."""

    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        return self._body


@contextmanager
def replaying(
    client: Any, trajectory: Trajectory, *, mode: ReplayMode = ReplayMode.STRICT
) -> Generator[Player]:
    """Replay ``trajectory``'s LLM calls through ``client`` inside the block.

    Detects the client: an Anthropic client (has ``messages``) or an OpenAI
    client (has ``chat``). Each call the agent makes returns the recorded
    response with no network. In strict mode (default) a changed request raises
    ``DivergenceError``; in lenient mode the drift is collected on the player's
    ``divergences`` instead. The original ``create`` is restored on exit, even if
    the block raises.
    """
    player = Player(trajectory, mode=mode)
    if hasattr(client, "messages"):
        owner: Any = client.messages
    elif hasattr(client, "chat"):
        owner = client.chat.completions
    else:
        raise TypeError(
            "reenact.replaying expects an Anthropic or OpenAI client; got "
            f"{type(client).__name__}"
        )

    original = owner.create
    had_own = "create" in vars(owner)

    def _wrapped(**request: Any) -> _ReplayedResponse:
        return _ReplayedResponse(player.replay_llm_call(request))

    owner.create = _wrapped
    try:
        yield player
    finally:
        if had_own:
            owner.create = original
        else:
            del owner.create
