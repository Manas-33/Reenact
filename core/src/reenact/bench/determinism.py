"""Determinism benchmark: replay the whole corpus offline and measure byte-identity.

For each recorded run, every captured call is replayed back through the engine and
must reproduce the recording with zero divergences, and the cassette must
re-serialize byte-for-byte. The rate is reported **segmented** sequential vs.
parallel-tool: unordered parallel windows are the harder case, so the headline
number is not allowed to hide behind the easy one.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reenact.replay import Player, ReplayMode
from reenact.schema import LLMCallEvent, ToolCallEvent, Trajectory
from reenact.store import load_cassette

CORPUS = Path(__file__).resolve().parents[3] / "examples" / "corpus"
FLOOR_PCT = 95.0


@dataclass
class Segment:
    """Byte-identical tally for one class of runs."""

    total: int = 0
    byte_identical: int = 0

    @property
    def pct(self) -> float:
        return round(100.0 * self.byte_identical / self.total, 2) if self.total else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "total": self.total,
            "byte_identical": self.byte_identical,
            "pct": self.pct,
        }


def _has_parallel_window(trajectory: Trajectory) -> bool:
    """True if any two tool calls share a parent - a genuine parallel window."""
    counts: dict[int, int] = {}
    for event in trajectory.events:
        if isinstance(event, ToolCallEvent) and event.parent_seq is not None:
            counts[event.parent_seq] = counts.get(event.parent_seq, 0) + 1
    return any(count > 1 for count in counts.values())


def _replays_clean(trajectory: Trajectory) -> bool:
    """Replay every recorded call through the engine; True if nothing diverged."""
    player = Player(trajectory, mode=ReplayMode.LENIENT)
    for event in trajectory.events:
        if isinstance(event, LLMCallEvent):
            player.replay_llm_call(event.request)
        elif isinstance(event, ToolCallEvent):
            player.replay_tool_call(event.name, event.arguments)
    return not player.divergences


def _round_trips(path: Path, trajectory: Trajectory) -> bool:
    """True if the loaded trajectory re-serializes to the exact bytes on disk."""
    reserialized = trajectory.model_dump_json(indent=2) + "\n"
    return reserialized == path.read_text(encoding="utf-8")


def measure_determinism(corpus: Path = CORPUS) -> dict[str, Any]:
    """Replay every recording in ``corpus`` and report the byte-identical rate."""
    segments = {"sequential": Segment(), "parallel": Segment()}
    for path in sorted(corpus.glob("*.json")):
        trajectory = load_cassette(path)
        key = "parallel" if _has_parallel_window(trajectory) else "sequential"
        segment = segments[key]
        segment.total += 1
        if _replays_clean(trajectory) and _round_trips(path, trajectory):
            segment.byte_identical += 1

    total = sum(s.total for s in segments.values())
    byte_identical = sum(s.byte_identical for s in segments.values())
    pct = round(100.0 * byte_identical / total, 2) if total else 0.0
    return {
        "metric": "replay_byte_identical_pct",
        "corpus": str(corpus),
        "total_runs": total,
        "byte_identical": byte_identical,
        "pct": pct,
        "floor_pct": FLOOR_PCT,
        "within_floor": total > 0 and pct >= FLOOR_PCT,
        "segments": {name: seg.as_dict() for name, seg in segments.items()},
    }
