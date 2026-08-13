"""The SQLite store persists, loads, lists, and round-trips trajectories."""

from pathlib import Path

import pytest

from reenact.record import Recorder
from reenact.schema import Trajectory
from reenact.store import TrajectoryStore


def _trajectory(name: str) -> Trajectory:
    rec = Recorder(name=name)
    rec.record_llm_call(
        provider="anthropic", model="m", request={"messages": []}, response={}
    )
    return rec.trajectory


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    with TrajectoryStore(tmp_path / "s.db") as store:
        traj = _trajectory("run-a")
        store.save(traj)
        assert store.load(traj.id) == traj


def test_persists_across_connections(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    traj = _trajectory("run-a")
    with TrajectoryStore(db) as store:
        store.save(traj)
    with TrajectoryStore(db) as store:
        assert store.load(traj.id).name == "run-a"


def test_list_and_missing(tmp_path: Path) -> None:
    with TrajectoryStore(tmp_path / "s.db") as store:
        store.save(_trajectory("a"))
        store.save(_trajectory("b"))
        names = {row["name"] for row in store.list_trajectories()}
        assert names == {"a", "b"}
        with pytest.raises(KeyError):
            store.load("nope")


def test_export_and_import_cassette(tmp_path: Path) -> None:
    traj = _trajectory("run-a")
    with TrajectoryStore(tmp_path / "s.db") as store:
        store.save(traj)
        store.export_cassette(traj.id, tmp_path / "run.json")
    with TrajectoryStore(tmp_path / "other.db") as store:
        assert store.import_cassette(tmp_path / "run.json") == traj
        assert store.load(traj.id) == traj
