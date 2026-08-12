"""Cassettes round-trip a trajectory through the filesystem, byte-stably."""

from pathlib import Path

from reenact.schema import LLMCallEvent, Trajectory
from reenact.store import load_cassette, save_cassette


def _trajectory() -> Trajectory:
    return Trajectory(
        name="sample",
        events=[
            LLMCallEvent(
                seq=0,
                provider="anthropic",
                model="claude-sonnet-4-5",
                request={"model": "claude-sonnet-4-5", "messages": []},
                response={"id": "msg_1", "content": []},
                request_hash="abc123",
            )
        ],
    )


def test_cassette_round_trips_a_trajectory(tmp_path: Path) -> None:
    traj = _trajectory()
    path = tmp_path / "run.json"
    save_cassette(traj, path)
    assert load_cassette(path) == traj


def test_cassette_is_deterministic(tmp_path: Path) -> None:
    traj = _trajectory()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    save_cassette(traj, first)
    save_cassette(traj, second)
    assert first.read_bytes() == second.read_bytes()
    assert first.read_text(encoding="utf-8").endswith("\n")
