"""Schema v0: trajectory and typed events validate and round-trip correctly."""

from __future__ import annotations

from reenact.schema import (
    SCHEMA_VERSION,
    GraphNodeEvent,
    LLMCallEvent,
    SideEffect,
    TokenUsage,
    ToolCallEvent,
    Trajectory,
)


def _sample_trajectory() -> Trajectory:
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
                usage=TokenUsage(input_tokens=10, output_tokens=5),
                cost_usd=0.0001,
                latency_ms=123.4,
            ),
            ToolCallEvent(
                seq=1,
                parent_seq=0,
                name="search_docs",
                arguments={"q": "reenact"},
                result={"hits": 3},
                side_effect=SideEffect.READ_ONLY,
            ),
            GraphNodeEvent(
                seq=2, node="agent", step=1, thread_id="t1", checkpoint_ns="agent:abc"
            ),
        ],
    )


def test_trajectory_stamps_version_and_id() -> None:
    traj = _sample_trajectory()
    assert traj.schema_version == SCHEMA_VERSION
    assert traj.id
    assert len(traj.events) == 3


def test_events_form_a_discriminated_union() -> None:
    traj = _sample_trajectory()
    llm, tool, node = traj.events
    assert isinstance(llm, LLMCallEvent)
    assert isinstance(tool, ToolCallEvent)
    assert isinstance(node, GraphNodeEvent)
    assert llm.request_hash == "abc123"
    assert tool.side_effect is SideEffect.READ_ONLY
    assert node.step == 1
    assert node.checkpoint_ns == "agent:abc"


def test_json_round_trip_rebuilds_event_types() -> None:
    traj = _sample_trajectory()
    restored = Trajectory.model_validate_json(traj.model_dump_json())
    assert restored.schema_version == traj.schema_version

    llm, tool, node = restored.events
    assert isinstance(llm, LLMCallEvent)
    assert isinstance(tool, ToolCallEvent)
    assert isinstance(node, GraphNodeEvent)
    # verbatim bodies and typed fields survive serialization
    assert llm.request == {"model": "claude-sonnet-4-5", "messages": []}
    assert llm.usage is not None
    assert llm.usage.input_tokens == 10
    assert tool.arguments == {"q": "reenact"}
    assert node.node == "agent"


def test_default_collections_are_independent() -> None:
    # Pydantic v2 copies mutable defaults per instance, so empty defaults are
    # never shared between trajectories.
    a = Trajectory()
    b = Trajectory()
    assert a.events is not b.events
    assert a.metadata is not b.metadata
