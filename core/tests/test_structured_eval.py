"""The structured trajectory evaluator - evidence-backed soft assertions.

The evaluator client is stubbed (deterministic, no key), so these prove the
mechanism: criteria parse to per-criterion verdicts, a pass with no or a
fabricated citation is downgraded to fail, every criterion answers in one batched
call, the soft assertions mix with hard ones in a scenario, the task-general
faithfulness criterion works, and a garbled evaluation fails closed. Calibrating
the criteria against human labels is a later rung.
"""

import json
from typing import Any

from reenact.evals import (
    FAITHFULNESS,
    Baseline,
    Criterion,
    CriterionLevel,
    PairwiseVerdict,
    Scenario,
    StructuredEvaluator,
    called_tool,
    pairwise,
    run_scenario,
    run_suite,
    structured_eval,
)
from reenact.evals.check import RunView
from reenact.evals.structured import STRUCTURED_SYSTEM
from reenact.schema import LLMCallEvent, SideEffect, ToolCallEvent, Trajectory


class _StubResponse:
    """Mimics an Anthropic response object: a text block behind ``model_dump``."""

    def __init__(self, text: str) -> None:
        self._text = text

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        return {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": self._text}],
        }


class _StubMessages:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _StubResponse:
        self.calls.append(kwargs)
        return _StubResponse(self._text)


class _StubClient:
    """An evaluator client whose reply is fixed, capturing the requests it gets."""

    def __init__(self, text: str) -> None:
        self.messages = _StubMessages(text)


def _verdicts_json(*verdicts: dict[str, Any]) -> str:
    return json.dumps(list(verdicts))


def _run_checks(client: Any, criteria: list[Criterion]) -> list[Any]:
    """Run every criterion check over one weather run and return the results."""
    view = RunView(_weather_run())
    return [check(view) for check in structured_eval(client, criteria)]


def _weather_run() -> Trajectory:
    question = [{"role": "user", "content": "What's the weather in Paris?"}]
    return Trajectory(
        name="weather",
        events=[
            LLMCallEvent(
                seq=0,
                provider="anthropic",
                model="m",
                request={"messages": question},
                response={
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "get_weather",
                            "input": {"city": "Paris"},
                        }
                    ]
                },
                request_hash="h0",
            ),
            ToolCallEvent(
                seq=1,
                name="get_weather",
                arguments={"city": "Paris"},
                result="18C and cloudy",
                side_effect=SideEffect.READ_ONLY,
            ),
            LLMCallEvent(
                seq=2,
                provider="anthropic",
                model="m",
                request={"messages": question},
                response={
                    "content": [
                        {"type": "text", "text": "It is 18C and cloudy in Paris."}
                    ]
                },
                request_hash="h2",
            ),
        ],
    )


# --- raw verdicts ------------------------------------------------------------


def test_criteria_parse_to_verdicts() -> None:
    client = _StubClient(
        _verdicts_json(
            {"id": "used_tool", "passed": True, "evidence": "[1]", "reasoning": "ok"},
            {"id": "grounded", "passed": False, "evidence": "", "reasoning": "no"},
        )
    )
    criteria = [
        Criterion(id="used_tool", question="Did it call the weather tool?"),
        Criterion(id="grounded", question="Is the answer grounded?"),
    ]
    verdicts = StructuredEvaluator(client, criteria).evaluate(RunView(_weather_run()))
    assert set(verdicts) == {"used_tool", "grounded"}
    assert verdicts["used_tool"].passed
    assert verdicts["used_tool"].evidence == "[1]"
    assert not verdicts["grounded"].passed


# --- prompt construction -----------------------------------------------------


def test_prompt_lists_criteria_and_transcript() -> None:
    client = _StubClient(
        _verdicts_json({"id": "used_tool", "passed": True, "evidence": "[1]"})
    )
    criteria = [Criterion(id="used_tool", question="Did it call the weather tool?")]
    structured_eval(client, criteria)[0](RunView(_weather_run()))
    call = client.messages.calls[0]
    assert call["system"] == STRUCTURED_SYSTEM
    assert call["temperature"] == 0.0
    content = call["messages"][0]["content"]
    assert "id=used_tool: Did it call the weather tool?" in content
    assert "get_weather" in content  # the rendered transcript is in the prompt
    assert "Final answer" in content


# --- per-criterion soft assertions -------------------------------------------


def test_checks_produce_per_criterion_results() -> None:
    client = _StubClient(
        _verdicts_json(
            {"id": "used_tool", "passed": True, "evidence": "[1] called get_weather"},
            {"id": "answered", "passed": True, "evidence": "[2] reported cloudy"},
        )
    )
    criteria = [
        Criterion(id="used_tool", question="Did it call the weather tool?"),
        Criterion(id="answered", question="Did it answer the question?"),
    ]
    results = _run_checks(client, criteria)
    assert [r.name for r in results] == ["criterion:used_tool", "criterion:answered"]
    assert all(r.passed for r in results)
    assert all(r.score is None for r in results)  # binary, no scalar


# --- evidence downgrade ------------------------------------------------------


def test_pass_without_evidence_downgrades_to_fail() -> None:
    client = _StubClient(
        _verdicts_json({"id": "grounded", "passed": True, "evidence": ""})
    )
    check = structured_eval(client, [Criterion(id="grounded", question="Grounded?")])[0]
    result = check(RunView(_weather_run()))
    assert not result.passed
    assert "downgraded to fail" in result.message


def test_pass_with_fabricated_citation_downgrades_to_fail() -> None:
    # The trajectory has steps [0], [1], [2]; a citation to [9] is hallucinated.
    client = _StubClient(
        _verdicts_json({"id": "grounded", "passed": True, "evidence": "see step [9]"})
    )
    check = structured_eval(client, [Criterion(id="grounded", question="Grounded?")])[0]
    result = check(RunView(_weather_run()))
    assert not result.passed
    assert "downgraded to fail" in result.message


def test_fail_verdict_needs_no_evidence() -> None:
    # A fail is the safe direction, so it is not evidence-gated (never upgraded).
    client = _StubClient(
        _verdicts_json({"id": "grounded", "passed": False, "evidence": ""})
    )
    check = structured_eval(client, [Criterion(id="grounded", question="Grounded?")])[0]
    result = check(RunView(_weather_run()))
    assert not result.passed
    assert "not satisfied" in result.message


# --- batching ----------------------------------------------------------------


def test_all_criteria_answered_in_one_call() -> None:
    client = _StubClient(
        _verdicts_json(
            {"id": "a", "passed": True, "evidence": "[0]"},
            {"id": "b", "passed": True, "evidence": "[1]"},
            {"id": "c", "passed": True, "evidence": "[2]"},
        )
    )
    criteria = [Criterion(id=cid, question="?") for cid in ("a", "b", "c")]
    scenario = Scenario(
        name="weather",
        trajectory=_weather_run(),
        checks=structured_eval(client, criteria),
    )
    result = run_scenario(scenario)
    assert result.passed
    assert len(result.checks) == 3
    # Three criteria, one model call - the memoized batch, not one call each.
    assert len(client.messages.calls) == 1


# --- integration -------------------------------------------------------------


def test_mixes_with_hard_assertions_in_a_scenario() -> None:
    client = _StubClient(
        _verdicts_json({"id": "grounded", "passed": True, "evidence": "[2]"})
    )
    scenario = Scenario(
        name="weather",
        trajectory=_weather_run(),
        checks=[
            called_tool("get_weather"),
            *structured_eval(client, [Criterion(id="grounded", question="Grounded?")]),
        ],
    )
    result = run_scenario(scenario)
    assert result.passed
    assert [c.name for c in result.checks] == [
        "called_tool('get_weather')",
        "criterion:grounded",
    ]


# --- blocking vs advisory levels ---------------------------------------------


def test_criterion_defaults_to_blocking() -> None:
    assert Criterion(id="x", question="?").level is CriterionLevel.BLOCKING
    result = structured_eval(
        _StubClient(_verdicts_json({"id": "x", "passed": True, "evidence": "[1]"})),
        [Criterion(id="x", question="?")],
    )[0](RunView(_weather_run()))
    assert result.level is CriterionLevel.BLOCKING


def test_advisory_criterion_carries_its_level() -> None:
    client = _StubClient(
        _verdicts_json({"id": "tone", "passed": False, "evidence": ""})
    )
    criterion = Criterion(
        id="tone", question="Polite tone?", level=CriterionLevel.ADVISORY
    )
    result = structured_eval(client, [criterion])[0](RunView(_weather_run()))
    # The level rides onto the CheckResult, so the gate can warn without blocking.
    assert result.level is CriterionLevel.ADVISORY
    assert not result.passed


def test_advisory_level_survives_into_the_baseline() -> None:
    # The full seam: criterion level -> CheckResult -> Baseline.from_report, so the
    # committed baseline records which checks only warn.
    client = _StubClient(
        _verdicts_json({"id": "tone", "passed": True, "evidence": "[1]"})
    )
    scenario = Scenario(
        name="weather",
        trajectory=_weather_run(),
        checks=structured_eval(
            client, [Criterion(id="tone", question="?", level=CriterionLevel.ADVISORY)]
        ),
    )
    baseline = Baseline.from_report(run_suite([scenario]))
    recorded = baseline.scenarios[0].checks[0]
    assert recorded.name == "criterion:tone"
    assert recorded.level is CriterionLevel.ADVISORY


def test_faithfulness_is_a_criterion() -> None:
    client = _StubClient(
        _verdicts_json(
            {"id": "faithful", "passed": True, "evidence": "[2] matches tool result"}
        )
    )
    check = structured_eval(client, [FAITHFULNESS])[0]
    result = check(RunView(_weather_run()))
    assert result.name == "criterion:faithful"
    assert result.passed


# --- robustness --------------------------------------------------------------


def test_missing_verdict_fails_closed() -> None:
    # The evaluator answers only one of two criteria; the unanswered one fails.
    client = _StubClient(
        _verdicts_json({"id": "a", "passed": True, "evidence": "[0]"})
    )
    criteria = [Criterion(id="a", question="?"), Criterion(id="b", question="?")]
    results = _run_checks(client, criteria)
    by_name = {r.name: r for r in results}
    assert by_name["criterion:a"].passed
    assert not by_name["criterion:b"].passed
    assert "no verdict" in by_name["criterion:b"].message


def test_garbled_reply_fails_every_criterion() -> None:
    client = _StubClient("totally not json")
    criteria = [Criterion(id="a", question="?"), Criterion(id="b", question="?")]
    results = _run_checks(client, criteria)
    assert not any(r.passed for r in results)
    assert all("no verdict" in r.message for r in results)


def test_verdicts_parse_when_wrapped_in_prose() -> None:
    client = _StubClient(
        'Here are my verdicts:\n```json\n'
        '[{"id": "a", "passed": true, "evidence": "[1]"}]\n```\nDone.'
    )
    check = structured_eval(client, [Criterion(id="a", question="?")])[0]
    assert check(RunView(_weather_run())).passed


# --- pairwise ----------------------------------------------------------------


def _pairwise(text: str) -> PairwiseVerdict:
    client = _StubClient(text)
    return pairwise(
        client,
        _weather_run(),
        _weather_run(),
        Criterion(id="quality", question="Which run triaged better?"),
    )


def test_pairwise_returns_worse_same_better() -> None:
    assert _pairwise('{"comparison": "worse", "reasoning": "B dropped a step"}') is (
        PairwiseVerdict.WORSE
    )
    assert _pairwise('{"comparison": "same", "reasoning": "no difference"}') is (
        PairwiseVerdict.SAME
    )
    assert _pairwise('{"comparison": "better", "reasoning": "B is clearer"}') is (
        PairwiseVerdict.BETTER
    )


def test_pairwise_prompt_shows_both_runs() -> None:
    client = _StubClient('{"comparison": "same"}')
    pairwise(
        client,
        _weather_run(),
        _weather_run(),
        Criterion(id="q", question="Which is better?"),
    )
    content = client.messages.calls[0]["messages"][0]["content"]
    assert "Run A (baseline)" in content
    assert "Run B (new)" in content
    assert "Which is better?" in content


def test_pairwise_fails_closed_to_worse() -> None:
    # A garbled or invalid comparison blocks (WORSE), never silently passes.
    assert _pairwise("not json at all") is PairwiseVerdict.WORSE
    assert _pairwise('{"comparison": "sideways"}') is PairwiseVerdict.WORSE
