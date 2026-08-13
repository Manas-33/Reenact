"""Parallel tool-call windows: concurrent calls match unordered within a window.

Tool calls a model fires together in one turn share a ``parent_seq`` and form an
unordered window - replaying them in a different order still matches. Calls with
no shared parent stay in exact recorded order.
"""

import pytest

from reenact.record import Recorder
from reenact.replay import DivergenceError, DivergenceKind, Player, ReplayMode
from reenact.schema import SideEffect


def _windowed_player(mode: ReplayMode = ReplayMode.STRICT) -> Player:
    # Three read-only tool calls issued in one model turn (shared parent) -> a
    # single unordered window.
    rec = Recorder()
    rec.record_llm_call(
        provider="anthropic", model="m", request={"a": 1}, response={"r": 1}
    )
    for city, result in [("Paris", "P"), ("London", "L")]:
        rec.record_tool_call(
            name="get_weather",
            arguments={"city": city},
            result=result,
            side_effect=SideEffect.READ_ONLY,
            parent_seq=0,
        )
    rec.record_tool_call(
        name="get_time",
        arguments={"tz": "UTC"},
        result="T",
        side_effect=SideEffect.READ_ONLY,
        parent_seq=0,
    )
    return Player(rec.trajectory, mode=mode)


def test_window_matches_calls_in_any_order() -> None:
    player = _windowed_player()
    # Replay in a different order than recorded - all still match.
    assert player.replay_tool_call("get_time", {"tz": "UTC"}) == "T"
    assert player.replay_tool_call("get_weather", {"city": "London"}) == "L"
    assert player.replay_tool_call("get_weather", {"city": "Paris"}) == "P"
    assert player.divergences == []


def test_window_consumes_each_call_once() -> None:
    player = _windowed_player()
    player.replay_tool_call("get_weather", {"city": "Paris"})
    player.replay_tool_call("get_weather", {"city": "London"})
    player.replay_tool_call("get_time", {"tz": "UTC"})
    # The window is used up; a further call has nothing left to match.
    with pytest.raises(DivergenceError) as excinfo:
        player.replay_tool_call("get_time", {"tz": "UTC"})
    assert excinfo.value.divergence.kind is DivergenceKind.EXHAUSTED


def test_call_not_in_window_diverges() -> None:
    player = _windowed_player()
    with pytest.raises(DivergenceError) as excinfo:
        player.replay_tool_call("delete_everything", {"scope": "all"})
    assert excinfo.value.divergence.kind is DivergenceKind.TOOL_CALL


def test_lenient_window_records_and_falls_back() -> None:
    player = _windowed_player(ReplayMode.LENIENT)
    # An unmatched call is recorded, not raised, and the window still drains.
    player.replay_tool_call("mystery", {})
    assert len(player.divergences) == 1
    assert player.divergences[0].kind is DivergenceKind.TOOL_CALL


def test_sequential_tools_without_a_shared_parent_stay_ordered() -> None:
    # Two tools with no shared parent_seq -> two windows of one -> exact order.
    rec = Recorder()
    rec.record_tool_call(name="a", arguments={}, result="A")
    rec.record_tool_call(name="b", arguments={}, result="B")
    player = Player(rec.trajectory)
    # Calling "b" first is out of order and must diverge.
    with pytest.raises(DivergenceError):
        player.replay_tool_call("b", {})


def test_window_still_never_re_fires_a_mutating_tool() -> None:
    fired: list[str] = []

    def spy() -> str:
        fired.append("SIDE EFFECT")
        return "live"

    rec = Recorder()
    rec.record_llm_call(
        provider="anthropic", model="m", request={"a": 1}, response={"r": 1}
    )
    for tool_id, ack in [(1, "ack1"), (2, "ack2")]:
        rec.record_tool_call(
            name="post_reply",
            arguments={"id": tool_id},
            result=ack,
            side_effect=SideEffect.MUTATING,
            parent_seq=0,
        )
    player = Player(rec.trajectory)
    # Out of order, and every real call would fire a side effect if run.
    assert player.replay_tool_call("post_reply", {"id": 2}, run=spy) == "ack2"
    assert player.replay_tool_call("post_reply", {"id": 1}, run=spy) == "ack1"
    assert fired == []  # unordered matching, and still nothing re-fired
