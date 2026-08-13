"""The substitution engine: answer a live run's calls from a recording.

Each live call is matched to the next recorded call *of the same type* - an
independent cursor per call type - and verified by fingerprint. A match returns
the recorded response or result with no network; a mismatch is a divergence:
raised in strict mode, collected in lenient mode, never a silently wrong answer.
"""

from enum import StrEnum
from typing import Any

from reenact.record import hash_request, redact
from reenact.replay.divergence import Divergence, DivergenceError, DivergenceKind
from reenact.schema import LLMCallEvent, ToolCallEvent, Trajectory


class ReplayMode(StrEnum):
    """How the player reacts to a call that does not match the recording.

    ``STRICT`` raises on the first divergence - the regression-test posture, drift
    fails loudly. ``LENIENT`` records it and returns the recorded value anyway, so
    a drifted run still plays through and every departure is collected for a
    regression diff to read. Neither mode fabricates a value for a call with no
    recorded counterpart - an exhausted recording always raises.
    """

    STRICT = "strict"
    LENIENT = "lenient"


def _tool_fingerprint(name: str, arguments: dict[str, Any]) -> str:
    """Fingerprint a tool call over its name and (redacted) arguments together.

    Folding the name and arguments into one hash means a changed tool name *or*
    changed arguments both diverge, uniform with the LLM request fingerprint.
    """
    payload: dict[str, Any] = {"name": name, "arguments": redact(arguments)}
    return hash_request(payload)


class Player:
    """Replays a recorded trajectory's calls, matched by (call type, sequence).

    LLM and tool calls each advance their own cursor, so an agent whose tool
    calls are not routed through reenact still replays its model calls in order.
    A fingerprint mismatch is a :class:`Divergence`: raised in strict mode,
    appended to :attr:`divergences` in lenient mode.
    """

    def __init__(
        self, trajectory: Trajectory, *, mode: ReplayMode = ReplayMode.STRICT
    ) -> None:
        self.mode = mode
        self.divergences: list[Divergence] = []
        self._llm_calls = [e for e in trajectory.events if isinstance(e, LLMCallEvent)]
        self._tool_calls = [
            e for e in trajectory.events if isinstance(e, ToolCallEvent)
        ]
        self._llm_cursor = 0
        self._tool_cursor = 0

    def replay_llm_call(self, request: dict[str, Any]) -> dict[str, Any]:
        """Return the recorded response for ``request``, matched by fingerprint."""
        if self._llm_cursor >= len(self._llm_calls):
            raise DivergenceError(
                Divergence(
                    kind=DivergenceKind.EXHAUSTED,
                    message="no recorded LLM call left to replay for this request",
                )
            )
        expected = self._llm_calls[self._llm_cursor]
        actual_hash = hash_request(redact(request))
        if actual_hash != expected.request_hash:
            self._report(
                Divergence(
                    kind=DivergenceKind.LLM_REQUEST,
                    seq=expected.seq,
                    expected=expected.request_hash,
                    actual=actual_hash,
                    message=(
                        f"LLM call at step {expected.seq} diverged: expected "
                        f"request {expected.request_hash}, got {actual_hash}"
                    ),
                )
            )
        self._llm_cursor += 1
        return expected.response

    def replay_tool_call(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> Any:
        """Return the recorded result for a tool call, matched by name + arguments.

        This is pure substitution - the recorded result is handed back and the
        real tool is never run. Whether a read-only tool *may* be re-executed live
        instead is the rung-2.2 side-effect policy layered on top of this.
        """
        if self._tool_cursor >= len(self._tool_calls):
            raise DivergenceError(
                Divergence(
                    kind=DivergenceKind.EXHAUSTED,
                    message=f"no recorded tool call left to replay for {name!r}",
                )
            )
        expected = self._tool_calls[self._tool_cursor]
        args = arguments if arguments is not None else {}
        actual_hash = _tool_fingerprint(name, args)
        expected_hash = _tool_fingerprint(expected.name, expected.arguments)
        if actual_hash != expected_hash:
            self._report(
                Divergence(
                    kind=DivergenceKind.TOOL_CALL,
                    seq=expected.seq,
                    expected=expected_hash,
                    actual=actual_hash,
                    message=(
                        f"tool call at step {expected.seq} diverged: expected "
                        f"{expected.name} {expected_hash}, got {name} {actual_hash}"
                    ),
                )
            )
        self._tool_cursor += 1
        return expected.result

    def _report(self, divergence: Divergence) -> None:
        """Raise in strict mode, collect in lenient mode."""
        if self.mode is ReplayMode.STRICT:
            raise DivergenceError(divergence)
        self.divergences.append(divergence)
