"""End-to-end: the callback handler records a real LangGraph run.

Unlike test_langchain.py (which drives the callbacks with synthetic metadata),
this builds an actual checkpointed graph and invokes it, so the node names and
checkpoint coordinates come from LangGraph itself - the check that our metadata
keys match reality.
"""

# LangGraph ships incomplete type stubs; run this file in basic mode so its
# partially-unknown generics don't trip strict checking of our own code.
# pyright: basic

from typing import Any, TypedDict

import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from reenact.record.langchain import ReenactCallbackHandler
from reenact.schema import GraphNodeEvent


class _State(TypedDict):
    value: int


def _increment(state: _State) -> _State:
    return {"value": state["value"] + 1}


def _double(state: _State) -> _State:
    return {"value": state["value"] * 2}


def _compiled_graph() -> Any:
    builder = StateGraph(_State)
    builder.add_node("increment", _increment)
    builder.add_node("double", _double)
    builder.add_edge(START, "increment")
    builder.add_edge("increment", "double")
    builder.add_edge("double", END)
    return builder.compile(checkpointer=MemorySaver())


def test_records_real_langgraph_node_boundaries() -> None:
    graph = _compiled_graph()
    handler = ReenactCallbackHandler()
    result = graph.invoke(
        {"value": 1},
        config={"callbacks": [handler], "configurable": {"thread_id": "t1"}},
    )
    assert result == {"value": 4}  # (1 + 1) * 2

    nodes = [
        event
        for event in handler.recorder.trajectory.events
        if isinstance(event, GraphNodeEvent)
    ]
    # Node names come straight from LangGraph, in execution order.
    assert [n.node for n in nodes] == ["increment", "double"]
    # Every boundary carries the checkpoint coordinates a fork resolves from.
    for node in nodes:
        assert node.thread_id == "t1"
        assert node.step is not None
        assert node.checkpoint_ns  # non-empty namespace like "increment:<uuid>"
    assert [n.step for n in nodes] == [1, 2]
