"""The recorder captures tool calls with their side-effect class."""

from reenact.record import REDACTED, Recorder
from reenact.schema import SideEffect


def test_records_a_tool_call() -> None:
    rec = Recorder()
    event = rec.record_tool_call(
        name="search_docs",
        arguments={"query": "reenact"},
        result={"hits": 3},
        side_effect=SideEffect.READ_ONLY,
    )
    assert event.name == "search_docs"
    assert event.arguments == {"query": "reenact"}
    assert event.result == {"hits": 3}
    assert event.side_effect is SideEffect.READ_ONLY


def test_tool_side_effect_defaults_to_unknown() -> None:
    rec = Recorder()
    event = rec.record_tool_call(name="do_thing")
    assert event.side_effect is SideEffect.UNKNOWN
    assert event.arguments == {}


def test_tool_and_llm_calls_share_one_ordered_trajectory() -> None:
    rec = Recorder()
    rec.record_llm_call(
        provider="anthropic", model="m", request={"messages": []}, response={}
    )
    rec.record_tool_call(name="t", result="ok")
    seqs = [e.seq for e in rec.trajectory.events]
    types = [e.type for e in rec.trajectory.events]
    assert seqs == [0, 1]
    assert types == ["llm_call", "tool_call"]


def test_tool_arguments_are_redacted() -> None:
    rec = Recorder()
    event = rec.record_tool_call(
        name="call_api",
        arguments={"url": "https://x", "api_key": "sk-secret"},
        result="ok",
    )
    assert event.arguments["api_key"] == REDACTED
    assert event.arguments["url"] == "https://x"
