"""Side-effect policy: substitute by default, re-run read-only only when opted in.

Includes the safety claim - a mutating tool's real function is never invoked on
replay - which is what makes replaying an agent safe to run anywhere.
"""

from reenact.record import Recorder
from reenact.replay import Player, ReplayPolicy
from reenact.schema import SideEffect


def _player_with_tool(
    effect: SideEffect, *, policy: ReplayPolicy | None = None
) -> Player:
    rec = Recorder()
    rec.record_tool_call(
        name="do_it", arguments={"x": 1}, result="recorded", side_effect=effect
    )
    return Player(rec.trajectory, policy=policy)


# --- policy decisions ---


def test_default_policy_substitutes_everything() -> None:
    policy = ReplayPolicy()
    assert policy.should_substitute("t", SideEffect.MUTATING)
    assert policy.should_substitute("t", SideEffect.UNKNOWN)
    assert policy.should_substitute("t", SideEffect.READ_ONLY)  # opt-in required


def test_read_only_reexecutes_only_when_opted_in() -> None:
    policy = ReplayPolicy(reexecute_read_only=True)
    assert not policy.should_substitute("t", SideEffect.READ_ONLY)
    assert policy.should_substitute("t", SideEffect.MUTATING)  # still substituted
    assert policy.should_substitute("t", SideEffect.UNKNOWN)  # unknown -> mutating


def test_override_reclassifies_by_name() -> None:
    policy = ReplayPolicy(
        overrides={"safe_read": SideEffect.READ_ONLY}, reexecute_read_only=True
    )
    # An override wins over the class recorded on the event.
    assert not policy.should_substitute("safe_read", SideEffect.UNKNOWN)
    assert policy.should_substitute("other", SideEffect.UNKNOWN)


def test_from_config_parses_plain_strings() -> None:
    policy = ReplayPolicy.from_config(
        {"post": "mutating", "search": "read_only"}, reexecute_read_only=True
    )
    assert policy.should_substitute("post", SideEffect.UNKNOWN)
    assert not policy.should_substitute("search", SideEffect.UNKNOWN)


# --- the safety claim: a mutating tool is never re-fired on replay ---


def test_mutating_tool_is_never_re_fired_on_replay() -> None:
    fired: list[str] = []

    def real_tool() -> str:
        fired.append("SIDE EFFECT")  # e.g. posts a reply, labels an issue
        return "live result"

    player = _player_with_tool(SideEffect.MUTATING)  # default policy
    result = player.replay_tool_call("do_it", {"x": 1}, run=real_tool)

    assert result == "recorded"  # the recorded result is substituted...
    assert fired == []  # ...and the real tool never ran. The safety claim.


def test_unknown_tool_is_also_never_re_fired() -> None:
    fired: list[str] = []

    def real_tool() -> str:
        fired.append("x")
        return "live"

    player = _player_with_tool(SideEffect.UNKNOWN)  # unknown == mutating (safe)
    assert player.replay_tool_call("do_it", {"x": 1}, run=real_tool) == "recorded"
    assert fired == []


def test_read_only_tool_reexecutes_live_when_opted_in() -> None:
    fired: list[str] = []

    def real_tool() -> str:
        fired.append("x")
        return "live result"

    player = _player_with_tool(
        SideEffect.READ_ONLY, policy=ReplayPolicy(reexecute_read_only=True)
    )
    result = player.replay_tool_call("do_it", {"x": 1}, run=real_tool)

    assert result == "live result"  # fresh live data returned...
    assert fired == ["x"]  # ...the read-only tool did run.


def test_read_only_substitutes_when_not_opted_in() -> None:
    fired: list[str] = []

    def real_tool() -> str:
        fired.append("x")
        return "live"

    player = _player_with_tool(SideEffect.READ_ONLY)  # default: no re-execution
    assert player.replay_tool_call("do_it", {"x": 1}, run=real_tool) == "recorded"
    assert fired == []
