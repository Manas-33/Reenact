"""A scenario: one recorded trajectory bound to the checks it must satisfy.

This is the unit the eval runner consumes and the CI gate diffs against a
baseline. Checks are runtime callables, so a scenario is a plain object (not a
serializable model); the committable suite config (a later rung) builds these
from strings.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from reenact.evals.check import Check
from reenact.schema import Trajectory
from reenact.store import load_cassette


def _no_checks() -> list[Check]:
    return []


@dataclass
class Scenario:
    """A named recording plus the ordered checks the run must satisfy."""

    name: str
    trajectory: Trajectory
    checks: list[Check] = field(default_factory=_no_checks)

    @classmethod
    def from_cassette(
        cls,
        path: str | Path,
        checks: Iterable[Check],
        *,
        name: str | None = None,
    ) -> "Scenario":
        """Load a scenario from a committed cassette file.

        The scenario name defaults to the recording's own name, then the file
        stem, so a suite reads sensibly without every scenario naming itself.
        """
        source = Path(path)
        trajectory = load_cassette(source)
        resolved = name or trajectory.name or source.stem
        return cls(name=resolved, trajectory=trajectory, checks=list(checks))
