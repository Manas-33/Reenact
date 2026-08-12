"""Read and write trajectory cassettes.

A cassette is a single trajectory serialized as deterministic JSON: stable
field order plus a trailing newline, so recordings committed to a repository
diff cleanly.
"""

from pathlib import Path

from reenact.schema import Trajectory


def save_cassette(trajectory: Trajectory, path: Path) -> None:
    """Write ``trajectory`` to ``path`` as a JSON cassette."""
    path.write_text(trajectory.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_cassette(path: Path) -> Trajectory:
    """Load the trajectory stored in the JSON cassette at ``path``."""
    return Trajectory.model_validate_json(path.read_text(encoding="utf-8"))
