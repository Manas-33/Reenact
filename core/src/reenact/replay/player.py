"""Replay recorded LLM calls without touching the network."""

from typing import Any

from reenact.record import hash_request, redact
from reenact.schema import LLMCallEvent, Trajectory


class DivergenceError(Exception):
    """Raised when a live call does not match the recorded trajectory."""


class Player:
    """Replays the LLM calls of a recorded trajectory, in order.

    Each incoming request is matched to the next recorded LLM call by sequence,
    then verified by request fingerprint. A match returns the recorded response
    with no network call; a fingerprint mismatch raises ``DivergenceError`` -
    the live run has drifted from the recording.
    """

    def __init__(self, trajectory: Trajectory) -> None:
        self._calls = [e for e in trajectory.events if isinstance(e, LLMCallEvent)]
        self._cursor = 0

    def replay_llm_call(self, request: dict[str, Any]) -> dict[str, Any]:
        """Return the recorded response for ``request``, or raise on divergence."""
        if self._cursor >= len(self._calls):
            raise DivergenceError(
                "no recorded LLM call left to replay for this request"
            )
        expected = self._calls[self._cursor]
        actual_hash = hash_request(redact(request))
        if actual_hash != expected.request_hash:
            raise DivergenceError(
                f"LLM call at step {expected.seq} diverged: expected request "
                f"{expected.request_hash}, got {actual_hash}"
            )
        self._cursor += 1
        return expected.response
