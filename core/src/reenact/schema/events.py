"""Typed events that make up a recorded trajectory.

Events form a discriminated union keyed on ``type``. Every event carries a
monotonic ``seq`` (its total order within a trajectory) and an optional
``parent_seq`` linking it to the event that spawned it, so parallel branches
such as concurrent tool calls can be represented without losing ordering.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SideEffect(StrEnum):
    """How a tool call affects the world.

    The replay side-effect policy substitutes anything not explicitly
    ``READ_ONLY``, so ``UNKNOWN`` is the safe default (treated as mutating).
    """

    READ_ONLY = "read_only"
    MUTATING = "mutating"
    UNKNOWN = "unknown"


class TokenUsage(BaseModel):
    """Token counts reported for a single model call."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int | None = None
    output_tokens: int | None = None


class _EventBase(BaseModel):
    """Fields shared by every event in a trajectory."""

    model_config = ConfigDict(extra="forbid")

    seq: int = Field(ge=0, description="Monotonic position within the trajectory.")
    parent_seq: int | None = Field(
        default=None,
        description="seq of the spawning event; None for a top-level event.",
    )
    started_at: datetime | None = None
    latency_ms: float | None = None


class LLMCallEvent(_EventBase):
    """A single LLM call, with verbatim request and response bodies."""

    type: Literal["llm_call"] = "llm_call"
    provider: str
    model: str
    request: dict[str, Any]
    response: dict[str, Any]
    request_hash: str = Field(
        description="Stable hash of the canonicalized request, for replay matching."
    )
    usage: TokenUsage | None = None
    cost_usd: float | None = None


class ToolCallEvent(_EventBase):
    """A single tool invocation made by the agent."""

    type: Literal["tool_call"] = "tool_call"
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    side_effect: SideEffect = SideEffect.UNKNOWN


class GraphNodeEvent(_EventBase):
    """A LangGraph node boundary crossed during a run.

    Records the node name plus the coordinates that address the checkpoint taken
    at that boundary: the thread, the superstep, and the checkpoint namespace.
    LangGraph does not expose the persisted checkpoint id to the callback layer -
    it lives only in the checkpointer's state history - so a fork resolves the
    concrete checkpoint from ``(thread_id, step, checkpoint_ns)`` against the
    checkpointer. That anchor is what makes a fork possible.
    """

    type: Literal["graph_node"] = "graph_node"
    node: str
    step: int | None = None
    thread_id: str | None = None
    checkpoint_ns: str | None = None


# Union of every event type. Pydantic selects the right model by the ``type``
# literal that is unique to each member.
type Event = LLMCallEvent | ToolCallEvent | GraphNodeEvent
