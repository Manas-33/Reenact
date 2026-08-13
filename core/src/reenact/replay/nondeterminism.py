"""Nondeterminism shims: make a replay byte-identical where the live run isn't.

An agent that stamps the wall clock or a random value into a request builds a
*different* request every run, so its fingerprint would never match the
recording. These injectable seams - a ``Clock`` and an ``Rng`` the agent reads
instead of the stdlib - record the values seen at capture time and replay them,
so the agent rebuilds the identical request offline. ``replay_stream`` does the
same for a streamed response: it re-emits the recorded chunks.
"""

from __future__ import annotations

import random
import time
from collections.abc import Iterator
from datetime import UTC, datetime, tzinfo
from typing import Any, cast

from reenact.replay.divergence import Divergence, DivergenceError, DivergenceKind
from reenact.schema import Trajectory

ENTROPY_KEY = "entropy"


class Clock:
    """A clock the agent reads instead of ``time.time()`` / ``datetime.now()``.

    Record mode (``Clock()``) returns the real time and logs each read; replay
    mode (``Clock.replaying(log)``) returns the logged values in order, so agent
    code that stamps the time into a request rebuilds the identical request.
    Reading past the recorded log is a divergence - the live run made more clock
    reads than were captured.
    """

    def __init__(self, log: list[float] | None = None) -> None:
        self._replay = log is not None
        self._values: list[float] = list(log) if log is not None else []
        self._cursor = 0

    @classmethod
    def replaying(cls, log: list[float]) -> Clock:
        """A clock that replays a recorded log of timestamps."""
        return cls(log=log)

    def time(self) -> float:
        """Seconds since the epoch - real and logged, or replayed from the log."""
        if self._replay:
            if self._cursor >= len(self._values):
                raise DivergenceError(
                    Divergence(
                        kind=DivergenceKind.EXHAUSTED,
                        message="clock read past the recorded log during replay",
                    )
                )
            value = self._values[self._cursor]
            self._cursor += 1
            return value
        value = time.time()
        self._values.append(value)
        return value

    def now(self, tz: tzinfo = UTC) -> datetime:
        """The current time as a timezone-aware ``datetime``."""
        return datetime.fromtimestamp(self.time(), tz=tz)

    @property
    def log(self) -> list[float]:
        """The timestamps seen so far, to persist into the trajectory."""
        return list(self._values)


class Rng:
    """A random source the agent reads instead of the stdlib ``random``.

    Seeded from real entropy at record time and re-seeded with the recorded seed
    at replay time, so the same numbers come back - one integer persisted, not a
    log. Reads go through a private ``random.Random``, never the global module.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed if seed is not None else random.Random().getrandbits(64)
        self._random = random.Random(self.seed)

    def random(self) -> float:
        """A float in [0.0, 1.0)."""
        return self._random.random()

    def randint(self, a: int, b: int) -> int:
        """An integer in [a, b]."""
        return self._random.randint(a, b)

    def token(self, nbits: int = 128) -> str:
        """A hex token (e.g. a request id / nonce), reproducible on replay."""
        return format(self._random.getrandbits(nbits), "x")


class _ReplayedChunk:
    """A recorded stream chunk, dressed up so agent code can read it like a live
    one (``chunk.model_dump()``) without an SDK dependency."""

    def __init__(self, body: Any) -> None:
        self._body = body

    def model_dump(self, *, mode: str = "python") -> Any:
        return self._body


def replay_stream(chunks: list[Any]) -> Iterator[_ReplayedChunk]:
    """Re-emit recorded stream chunks as an iterator, offline.

    A streamed call is recorded as its list of chunk bodies; replaying it yields
    them back in order so streaming agent code - which accumulates chunks itself -
    reproduces the identical result with no network.
    """
    for body in chunks:
        yield _ReplayedChunk(body)


def reassemble_text(chunks: list[Any]) -> str:
    """Fold a recorded stream's text deltas into the full message text.

    Handles the common ``{"delta": {"text": ...}}`` chunk shape (Anthropic
    text_delta events); chunks without a text delta are skipped.
    """
    parts: list[str] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        delta = cast(dict[str, Any], chunk).get("delta")
        if isinstance(delta, dict):
            text = cast(dict[str, Any], delta).get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def save_entropy(
    trajectory: Trajectory, *, clock: Clock | None = None, rng: Rng | None = None
) -> None:
    """Persist a run's clock log and/or rng seed into the trajectory metadata."""
    entry: dict[str, Any] = {}
    if clock is not None:
        entry["clock"] = clock.log
    if rng is not None:
        entry["rng_seed"] = rng.seed
    trajectory.metadata[ENTROPY_KEY] = entry


def _entropy(trajectory: Trajectory) -> dict[str, Any]:
    entry = trajectory.metadata.get(ENTROPY_KEY)
    return cast(dict[str, Any], entry) if isinstance(entry, dict) else {}


def load_clock(trajectory: Trajectory) -> Clock:
    """A replay ``Clock`` seeded from the trajectory's recorded timestamps."""
    raw = _entropy(trajectory).get("clock")
    log = [float(v) for v in cast(list[Any], raw)] if isinstance(raw, list) else []
    return Clock.replaying(log)


def load_rng(trajectory: Trajectory) -> Rng:
    """A replay ``Rng`` seeded from the trajectory's recorded seed."""
    raw = _entropy(trajectory).get("rng_seed")
    return Rng(seed=int(raw) if isinstance(raw, int) else None)
