"""The trajectory: one full recorded agent run."""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from reenact.schema.events import Event

# Current on-disk schema version. Bumped on any breaking change; the loader
# migrates trajectories written under an older version.
SCHEMA_VERSION = 0


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


class Trajectory(BaseModel):
    """An ordered sequence of events captured from a single agent run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: uuid4().hex)
    name: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    events: list[Event] = []
    metadata: dict[str, Any] = {}
