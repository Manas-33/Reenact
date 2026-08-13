"""The substitution engine: matching, strict vs. lenient, and structured drift."""

from typing import Any

import pytest

from reenact.record import Recorder
from reenact.replay import (
    DivergenceError,
    DivergenceKind,
    Player,
    ReplayMode,
)
from reenact.schema import SideEffect

REQUEST: dict[str, Any] = {
    "model": "claude-sonnet-4-5",
    "messages": [{"role": "user", "content": "hi"}],
}
RESPONSE: dict[str, Any] = {
    "id": "msg_1",
    "content": [{"type": "text", "text": "Hello!"}],
}
CHANGED: dict[str, Any] = {
    "model": "claude-sonnet-4-5",
    "messages": [{"role": "user", "content": "different"}],
}


def _llm_player(mode: ReplayMode = ReplayMode.STRICT) -> Player:
    rec = Recorder(name="run")
    rec.record_llm_call(
        provider="anthropic",
        model="claude-sonnet-4-5",
        request=REQUEST,
        response=RESPONSE,
    )
    return Player(rec.trajectory, mode=mode)


def _tool_player(mode: ReplayMode = ReplayMode.STRICT) -> Player:
    rec = Recorder(name="run")
    rec.record_tool_call(
        name="get_weather",
        arguments={"city": "Paris"},
        result="18C and cloudy",
        side_effect=SideEffect.READ_ONLY,
    )
    return Player(rec.trajectory, mode=mode)


# --- strict LLM matching ---


def test_strict_matches_and_returns_recorded_response() -> None:
    player = _llm_player()
    assert player.replay_llm_call(REQUEST) == RESPONSE
    assert player.divergences == []


def test_strict_raises_structured_divergence_on_changed_request() -> None:
    player = _llm_player()
    with pytest.raises(DivergenceError) as excinfo:
        player.replay_llm_call(CHANGED)
    div = excinfo.value.divergence
    assert div.kind is DivergenceKind.LLM_REQUEST
    assert div.seq == 0
    assert div.expected is not None
    assert div.actual is not None
    assert div.expected != div.actual


# --- lenient LLM matching ---


def test_lenient_records_divergence_and_returns_recorded_response() -> None:
    player = _llm_player(ReplayMode.LENIENT)
    # A changed request does not raise; it plays through the recorded response...
    assert player.replay_llm_call(CHANGED) == RESPONSE
    # ...while the drift is captured for a regression diff to read.
    assert len(player.divergences) == 1
    assert player.divergences[0].kind is DivergenceKind.LLM_REQUEST


# --- exhaustion (both modes raise, nothing to substitute) ---


@pytest.mark.parametrize("mode", [ReplayMode.STRICT, ReplayMode.LENIENT])
def test_exhausted_recording_always_raises(mode: ReplayMode) -> None:
    player = _llm_player(mode)
    assert player.replay_llm_call(REQUEST) == RESPONSE
    with pytest.raises(DivergenceError) as excinfo:
        player.replay_llm_call(REQUEST)
    assert excinfo.value.divergence.kind is DivergenceKind.EXHAUSTED


# --- tool matching ---


def test_tool_call_replays_recorded_result() -> None:
    player = _tool_player()
    assert player.replay_tool_call("get_weather", {"city": "Paris"}) == "18C and cloudy"


def test_tool_call_diverges_on_changed_arguments() -> None:
    player = _tool_player()
    with pytest.raises(DivergenceError) as excinfo:
        player.replay_tool_call("get_weather", {"city": "London"})
    assert excinfo.value.divergence.kind is DivergenceKind.TOOL_CALL


def test_tool_call_diverges_on_changed_name() -> None:
    player = _tool_player()
    with pytest.raises(DivergenceError):
        player.replay_tool_call("get_forecast", {"city": "Paris"})


def test_lenient_tool_call_records_and_substitutes() -> None:
    player = _tool_player(ReplayMode.LENIENT)
    result = player.replay_tool_call("get_weather", {"city": "London"})
    assert result == "18C and cloudy"
    assert len(player.divergences) == 1
    assert player.divergences[0].kind is DivergenceKind.TOOL_CALL


# --- per-type cursors: llm and tool matching are independent ---


def test_llm_and_tool_cursors_are_independent() -> None:
    # A think -> act -> think trajectory: two llm calls around one tool call.
    rec = Recorder(name="run")
    rec.record_llm_call(
        provider="anthropic", model="m", request=REQUEST, response=RESPONSE
    )
    rec.record_tool_call(name="get_weather", arguments={"city": "Paris"}, result="18C")
    rec.record_llm_call(
        provider="anthropic", model="m", request=CHANGED, response={"id": "msg_2"}
    )
    player = Player(rec.trajectory)
    # The tool call replays regardless of the interleaved llm calls...
    assert player.replay_tool_call("get_weather", {"city": "Paris"}) == "18C"
    # ...and the two llm calls still match in their own recorded order.
    assert player.replay_llm_call(REQUEST) == RESPONSE
    assert player.replay_llm_call(CHANGED) == {"id": "msg_2"}
