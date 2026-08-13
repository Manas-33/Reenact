"""The eval runner core: assertions over a recording, aggregated by the runner.

Scenarios here are synthetic (each isolates one behavior), plus one run over a
real committed cassette to prove the runner works offline on real recordings.
"""

from pathlib import Path
from typing import Any

from reenact.evals import (
    Scenario,
    answer_contains,
    answer_matches,
    called_tool,
    did_not_call_tool,
    no_mutating_tool_reexecuted,
    replays_clean,
    run_scenario,
    run_suite,
    tool_call_count,
)
from reenact.evals.check import CheckResult, RunView
from reenact.record import hash_request, redact
from reenact.replay import ReplayPolicy
from reenact.schema import (
    LLMCallEvent,
    SideEffect,
    ToolCallEvent,
    Trajectory,
)

CORPUS = Path(__file__).resolve().parent.parent / "examples" / "corpus"


def _text_response(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
    }


def _tool_use_response(
    tool_id: str, name: str, tool_input: dict[str, Any]
) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}
        ],
    }


def _llm(seq: int, request: dict[str, Any], response: dict[str, Any]) -> LLMCallEvent:
    return LLMCallEvent(
        seq=seq,
        provider="anthropic",
        model="claude-sonnet-4-5",
        request=request,
        response=response,
        request_hash=hash_request(redact(request)),
    )


def _tool(
    seq: int,
    name: str,
    arguments: dict[str, Any],
    result: Any,
    side_effect: SideEffect = SideEffect.READ_ONLY,
    parent_seq: int | None = None,
) -> ToolCallEvent:
    return ToolCallEvent(
        seq=seq,
        parent_seq=parent_seq,
        name=name,
        arguments=arguments,
        result=result,
        side_effect=side_effect,
    )


def _weather_run() -> Trajectory:
    """A think -> act -> think recording: ask weather, call get_weather, answer."""
    question = [{"role": "user", "content": "What's the weather in Paris?"}]
    return Trajectory(
        name="weather",
        events=[
            _llm(
                0,
                {"messages": question},
                _tool_use_response("t1", "get_weather", {"city": "Paris"}),
            ),
            _tool(1, "get_weather", {"city": "Paris"}, "18C and cloudy"),
            _llm(
                2,
                {"messages": question, "step": 2},
                _text_response("It is 18C and cloudy in Paris."),
            ),
        ],
    )


# --- final-answer assertions -------------------------------------------------


def test_answer_contains_pass_and_fail() -> None:
    view = RunView(_weather_run())
    assert answer_contains("cloudy")(view).passed
    assert answer_contains("CLOUDY")(view).passed  # case-insensitive by default
    miss = answer_contains("snow")(view)
    assert not miss.passed
    assert "snow" in miss.message


def test_answer_contains_case_sensitive() -> None:
    view = RunView(_weather_run())
    assert not answer_contains("CLOUDY", case_sensitive=True)(view).passed
    assert answer_contains("cloudy", case_sensitive=True)(view).passed


def test_answer_matches_regex() -> None:
    view = RunView(_weather_run())
    assert answer_matches(r"\d+C")(view).passed
    assert not answer_matches(r"\d+F")(view).passed


def test_final_answer_reads_last_text_bearing_call() -> None:
    # The last event is a tool_use turn with no text; the answer is the prior text.
    traj = _weather_run()
    traj.events.append(
        _llm(
            3,
            {"messages": [{"role": "user", "content": "again"}]},
            _tool_use_response("t2", "get_weather", {"city": "Paris"}),
        )
    )
    assert RunView(traj).final_answer == "It is 18C and cloudy in Paris."


def test_final_answer_openai_shape() -> None:
    traj = Trajectory(
        events=[
            LLMCallEvent(
                seq=0,
                provider="openai",
                model="gpt-x",
                request={"messages": []},
                response={
                    "choices": [
                        {"message": {"role": "assistant", "content": "hello there"}}
                    ]
                },
                request_hash="h",
            )
        ]
    )
    assert RunView(traj).final_answer == "hello there"


# --- tool assertions ---------------------------------------------------------


def test_called_and_not_called_tool() -> None:
    view = RunView(_weather_run())
    assert called_tool("get_weather")(view).passed
    assert not called_tool("post_reply")(view).passed
    assert did_not_call_tool("post_reply")(view).passed
    assert not did_not_call_tool("get_weather")(view).passed


def test_tool_call_count() -> None:
    view = RunView(_weather_run())
    assert tool_call_count("get_weather", 1)(view).passed
    fail = tool_call_count("get_weather", 2)(view)
    assert not fail.passed
    assert "got 1" in fail.message


# --- replay-tied checks ------------------------------------------------------


def test_replays_clean_passes_on_consistent_recording() -> None:
    assert replays_clean()(RunView(_weather_run())).passed


def test_replays_clean_fails_on_tampered_hash() -> None:
    traj = _weather_run()
    # Corrupt a stored request hash so the recomputed one no longer matches.
    first = traj.events[0]
    assert isinstance(first, LLMCallEvent)
    first.request_hash = "sha256:tampered"
    result = replays_clean()(RunView(traj))
    assert not result.passed
    assert "divergence" in result.message


def test_no_mutating_tool_reexecuted_default_policy_passes() -> None:
    traj = _weather_run()
    traj.events.append(
        _tool(3, "post_reply", {"body": "hi"}, "posted", SideEffect.MUTATING)
    )
    assert no_mutating_tool_reexecuted()(RunView(traj)).passed


def test_no_mutating_tool_reexecuted_flags_dangerous_override() -> None:
    traj = _weather_run()
    traj.events.append(
        _tool(3, "post_reply", {"body": "hi"}, "posted", SideEffect.MUTATING)
    )
    # An override that reclassifies a mutating tool read-only AND opts into
    # re-execution would re-fire it live - exactly what the check must catch.
    danger = ReplayPolicy(
        overrides={"post_reply": SideEffect.READ_ONLY}, reexecute_read_only=True
    )
    result = no_mutating_tool_reexecuted(danger)(RunView(traj))
    assert not result.passed
    assert "post_reply" in result.message


def test_no_mutating_tool_reexecuted_allows_read_only_reexecution() -> None:
    # A genuinely read-only tool opted into re-execution is not a violation.
    policy = ReplayPolicy(reexecute_read_only=True)
    assert no_mutating_tool_reexecuted(policy)(RunView(_weather_run())).passed


# --- runner aggregation ------------------------------------------------------


def test_run_scenario_aggregates_checks() -> None:
    scenario = Scenario(
        name="weather",
        trajectory=_weather_run(),
        checks=[called_tool("get_weather"), answer_contains("cloudy"), replays_clean()],
    )
    result = run_scenario(scenario)
    assert result.passed
    assert len(result.checks) == 3
    assert not result.failures


def test_run_scenario_reports_failures() -> None:
    scenario = Scenario(
        name="weather",
        trajectory=_weather_run(),
        checks=[called_tool("get_weather"), answer_contains("snow")],
    )
    result = run_scenario(scenario)
    assert not result.passed
    assert [c.name for c in result.failures] == ["answer_contains('snow')"]


def test_run_suite_counts_pass_and_fail() -> None:
    good = Scenario("good", _weather_run(), [answer_contains("Paris")])
    bad = Scenario("bad", _weather_run(), [answer_contains("Berlin")])
    report = run_suite([good, bad])
    assert report.total == 2
    assert report.passed_count == 1
    assert not report.passed


def test_check_result_is_serializable() -> None:
    result = called_tool("get_weather")(RunView(_weather_run()))
    loaded = CheckResult.model_validate_json(result.model_dump_json())
    assert loaded == result


# --- over a real recording ---------------------------------------------------


def test_scenario_over_real_cassette_offline() -> None:
    scenario = Scenario.from_cassette(
        CORPUS / "tool_use-00.json",
        checks=[
            called_tool("get_weather"),
            answer_contains("cloudy"),
            replays_clean(),
            no_mutating_tool_reexecuted(),
        ],
    )
    result = run_scenario(scenario)
    assert result.passed, result.failures
    assert scenario.name == "tool_use"
