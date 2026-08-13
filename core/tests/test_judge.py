"""The trajectory-level LLM judge - a graded check over a whole run.

The judge client is stubbed (deterministic, no key), so these prove the
mechanism: the prompt is built from the trajectory, a score parses and gates on a
threshold, a reply wrapped in prose still parses, and a malformed or out-of-range
reply fails the check instead of raising. Calibrating the judge against human
labels is a later rung.
"""

from typing import Any

from reenact.evals import (
    Scenario,
    called_tool,
    judged,
    render_trajectory,
    run_scenario,
)
from reenact.evals.check import RunView
from reenact.evals.judge import JUDGE_SYSTEM
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
    """A judge client whose reply text is fixed, capturing the requests it gets."""

    def __init__(self, text: str) -> None:
        self.messages = _StubMessages(text)


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


# --- transcript rendering ----------------------------------------------------


def test_render_trajectory_is_multi_step() -> None:
    rendered = render_trajectory(_weather_run())
    assert "Task: What's the weather in Paris?" in rendered
    # The tool_use turn is shown as an action, the tool call with its result,
    # and the final answer - the multi-step view, not just the completion.
    assert "calls get_weather" in rendered
    assert 'tool get_weather({"city": "Paris"}) -> 18C and cloudy' in rendered
    assert "Final answer: It is 18C and cloudy in Paris." in rendered


# --- prompt construction -----------------------------------------------------


def test_judge_builds_prompt_from_trajectory() -> None:
    client = _StubClient('{"score": 0.9, "reasoning": "good"}')
    rubric = "The agent reports the weather using the tool."
    judged(client, rubric)(RunView(_weather_run()))
    call = client.messages.calls[0]
    assert call["system"] == JUDGE_SYSTEM
    content = call["messages"][0]["content"]
    assert rubric in content
    assert "get_weather" in content
    assert "Final answer" in content


# --- scoring and threshold ---------------------------------------------------


def test_judge_scores_and_passes_threshold() -> None:
    client = _StubClient('{"score": 0.9, "reasoning": "correct tool use"}')
    result = judged(client, "rubric", threshold=0.7)(RunView(_weather_run()))
    assert result.passed
    assert result.score == 0.9
    assert "correct tool use" in result.message


def test_judge_fails_below_threshold() -> None:
    client = _StubClient('{"score": 0.3, "reasoning": "wrong city"}')
    result = judged(client, "rubric", threshold=0.7)(RunView(_weather_run()))
    assert not result.passed
    assert result.score == 0.3


def test_judge_parses_json_wrapped_in_prose() -> None:
    client = _StubClient(
        'Here is my verdict:\n```json\n{"score": 0.8, "reasoning": "ok"}\n```'
    )
    result = judged(client, "rubric")(RunView(_weather_run()))
    assert result.passed
    assert result.score == 0.8


# --- robustness --------------------------------------------------------------


def test_judge_handles_malformed_reply() -> None:
    client = _StubClient("totally not json")
    result = judged(client, "rubric")(RunView(_weather_run()))
    assert not result.passed
    assert result.score is None
    assert "could not parse" in result.message


def test_judge_handles_out_of_range_score() -> None:
    client = _StubClient('{"score": 5, "reasoning": "too high"}')
    result = judged(client, "rubric")(RunView(_weather_run()))
    assert not result.passed
    assert result.score is None


# --- integration with the runner ---------------------------------------------


def test_judge_mixes_with_assertions_in_a_scenario() -> None:
    client = _StubClient('{"score": 0.85, "reasoning": "correct"}')
    scenario = Scenario(
        name="weather",
        trajectory=_weather_run(),
        checks=[
            called_tool("get_weather"),
            judged(client, "Weather reported correctly.", name="weather-quality"),
        ],
    )
    result = run_scenario(scenario)
    assert result.passed
    assert [c.name for c in result.checks] == [
        "called_tool('get_weather')",
        "weather-quality",
    ]
    assert result.checks[1].score == 0.85
