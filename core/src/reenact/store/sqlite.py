"""A zero-config SQLite store for recorded trajectories."""

import sqlite3
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from reenact.schema import Trajectory
from reenact.store.cassette import load_cassette, save_cassette

_DEFAULT_PATH = Path(".reenact") / "trajectories.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trajectories (
    id TEXT PRIMARY KEY,
    name TEXT,
    created_at TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    data TEXT NOT NULL
)
"""


class TrajectoryStore:
    """A zero-config SQLite store where recorded trajectories accumulate.

    The canonical local store, keyed by trajectory id; individual scenarios
    export to and import from git-friendly cassettes.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        db_path = Path(path) if path is not None else _DEFAULT_PATH
        if db_path.parent != Path():
            db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def save(self, trajectory: Trajectory) -> None:
        """Insert or replace ``trajectory`` by id."""
        self._conn.execute(
            "INSERT OR REPLACE INTO trajectories "
            "(id, name, created_at, schema_version, data) VALUES (?, ?, ?, ?, ?)",
            (
                trajectory.id,
                trajectory.name,
                trajectory.created_at.isoformat(),
                trajectory.schema_version,
                trajectory.model_dump_json(),
            ),
        )
        self._conn.commit()

    def load(self, trajectory_id: str) -> Trajectory:
        """Load a trajectory by id, or raise ``KeyError`` if it is not stored."""
        row = self._conn.execute(
            "SELECT data FROM trajectories WHERE id = ?", (trajectory_id,)
        ).fetchone()
        if row is None:
            raise KeyError(trajectory_id)
        return Trajectory.model_validate_json(row[0])

    def list_trajectories(self) -> list[dict[str, Any]]:
        """Return id/name/created_at/schema_version for every stored trajectory."""
        rows = self._conn.execute(
            "SELECT id, name, created_at, schema_version FROM trajectories "
            "ORDER BY created_at"
        ).fetchall()
        return [
            {"id": r[0], "name": r[1], "created_at": r[2], "schema_version": r[3]}
            for r in rows
        ]

    def export_cassette(self, trajectory_id: str, path: Path) -> None:
        """Write a stored trajectory out to a git-friendly cassette file."""
        save_cassette(self.load(trajectory_id), path)

    def import_cassette(self, path: Path) -> Trajectory:
        """Load a cassette file into the store and return the trajectory."""
        trajectory = load_cassette(path)
        self.save(trajectory)
        return trajectory

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
