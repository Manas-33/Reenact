"""The substitution engine: answer a live run's calls from a recording.

Each live call is matched to the next recorded call *of the same type* - an
independent cursor per call type - and verified by fingerprint. A match returns
the recorded response or result with no network; a mismatch is a divergence:
raised in strict mode, collected in lenient mode, never a silently wrong answer.
"""

from collections.abc import Callable
from enum import StrEnum
from typing import Any

from reenact.record import hash_request, redact
from reenact.replay.divergence import Divergence, DivergenceError, DivergenceKind
from reenact.replay.policy import ReplayPolicy
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


def _group_windows(tool_calls: list[ToolCallEvent]) -> list[list[ToolCallEvent]]:
    """Group tool calls into unordered windows of concurrent siblings.

    Consecutive tool calls that share a non-``None`` ``parent_seq`` (spawned by the
    same event, e.g. several tool_use blocks from one model turn) form one window,
    matched unordered. A tool call with no ``parent_seq`` is its own window of one,
    so ordinary sequential calls stay in exact order.
    """
    windows: list[list[ToolCallEvent]] = []
    for event in tool_calls:
        if (
            windows
            and event.parent_seq is not None
            and event.parent_seq == windows[-1][0].parent_seq
        ):
            windows[-1].append(event)
        else:
            windows.append([event])
    return windows


class Player:
    """Replays a recorded trajectory's calls, matched by (call type, sequence).

    LLM and tool calls each advance their own cursor, so an agent whose tool
    calls are not routed through reenact still replays its model calls in order.
    A fingerprint mismatch is a :class:`Divergence`: raised in strict mode,
    appended to :attr:`divergences` in lenient mode.
    """

    def __init__(
        self,
        trajectory: Trajectory,
        *,
        mode: ReplayMode = ReplayMode.STRICT,
        policy: ReplayPolicy | None = None,
    ) -> None:
        self.mode = mode
        self.policy = policy if policy is not None else ReplayPolicy()
        self.divergences: list[Divergence] = []
        self._llm_calls = [e for e in trajectory.events if isinstance(e, LLMCallEvent)]
        self._llm_cursor = 0
        tool_calls = [e for e in trajectory.events if isinstance(e, ToolCallEvent)]
        self._tool_windows = _group_windows(tool_calls)
        self._window_idx = 0
        self._window_remaining: list[ToolCallEvent] = []

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
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        run: Callable[[], Any] | None = None,
    ) -> Any:
        """Return the recorded result for a tool call - or re-run the real tool.

        Calls are matched within a *window* of recorded tool calls that ran
        concurrently (siblings sharing a ``parent_seq``): the incoming call is
        matched to any unconsumed member of the current window by fingerprint, so
        concurrent calls replayed in a different order still match. A window of one
        (the common case) is exact-order matching. On a match the recorded result
        is substituted and ``run`` is never called - unless the policy classifies
        the tool read-only with re-execution opted in, when the real tool runs and
        its fresh result is returned instead.
        """
        matched = self._match_tool_call(name, arguments)
        if run is not None and not self.policy.should_substitute(
            name, matched.side_effect
        ):
            return run()
        return matched.result

    def _match_tool_call(
        self, name: str, arguments: dict[str, Any] | None
    ) -> ToolCallEvent:
        """Consume and return the recorded tool call this live one matches.

        Advances to the next window when the current one is used up; a call that
        matches no unconsumed member of the window is a divergence (strict raises;
        lenient records it and falls back to the first unconsumed call).
        """
        if not self._window_remaining:
            if self._window_idx >= len(self._tool_windows):
                raise DivergenceError(
                    Divergence(
                        kind=DivergenceKind.EXHAUSTED,
                        message=f"no recorded tool call left to replay for {name!r}",
                    )
                )
            self._window_remaining = list(self._tool_windows[self._window_idx])
            self._window_idx += 1
        args = arguments if arguments is not None else {}
        actual_hash = _tool_fingerprint(name, args)
        for event in self._window_remaining:
            if _tool_fingerprint(event.name, event.arguments) == actual_hash:
                self._window_remaining.remove(event)
                return event
        expected = ", ".join(
            _tool_fingerprint(e.name, e.arguments) for e in self._window_remaining
        )
        fallback = self._window_remaining[0]
        names = [e.name for e in self._window_remaining]
        self._report(
            Divergence(
                kind=DivergenceKind.TOOL_CALL,
                seq=fallback.seq,
                expected=expected,
                actual=actual_hash,
                message=(
                    f"tool call {name} {actual_hash} matched no unconsumed recorded "
                    f"call in the window at step {fallback.seq} (remaining: {names})"
                ),
            )
        )
        self._window_remaining.remove(fallback)
        return fallback

    def _report(self, divergence: Divergence) -> None:
        """Raise in strict mode, collect in lenient mode."""
        if self.mode is ReplayMode.STRICT:
            raise DivergenceError(divergence)
        self.divergences.append(divergence)
