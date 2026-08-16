"""The regression gate: baseline diff plus the `reenact ci` verb.

The diff is tested directly on constructed baselines (each case isolates one kind
of change); the CLI tests write a baseline from a suite and then re-run `ci` after
mutating the recording to seed a regression - all offline.
"""

from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from reenact.cli import app
from reenact.evals import Baseline, CriterionLevel, RegressionDiff, diff_baselines
from reenact.evals.baseline import BaselineCheck, BaselineScenario
from reenact.record import hash_request, redact
from reenact.schema import LLMCallEvent, SideEffect, ToolCallEvent, Trajectory
from reenact.store import save_cassette

runner = CliRunner()


def _check(name: str, passed: bool, score: float | None = None) -> BaselineCheck:
    return BaselineCheck(name=name, passed=passed, score=score)


def _advisory(name: str, passed: bool) -> BaselineCheck:
    return BaselineCheck(name=name, passed=passed, level=CriterionLevel.ADVISORY)


def _scenario(name: str, *checks: BaselineCheck) -> BaselineScenario:
    return BaselineScenario(name=name, checks=list(checks))


def _baseline(*scenarios: BaselineScenario) -> Baseline:
    return Baseline(scenarios=list(scenarios))


# --- diff semantics ----------------------------------------------------------


def test_identical_run_has_no_regression() -> None:
    base = _baseline(_scenario("s", _check("c", True, 0.9)))
    diff = diff_baselines(base, base)
    assert not diff.regressed
    assert "no regressions" in diff.summary()


def test_pass_to_fail_is_a_regression() -> None:
    base = _baseline(_scenario("s", _check("called_tool('x')", True)))
    now = _baseline(_scenario("s", _check("called_tool('x')", False)))
    diff = diff_baselines(base, now)
    assert diff.regressed
    assert diff.regressions[0].detail == "pass->fail"


def test_score_drop_is_a_regression() -> None:
    base = _baseline(_scenario("s", _check("judge", True, 0.91)))
    now = _baseline(_scenario("s", _check("judge", False, 0.62)))
    diff = diff_baselines(base, now)
    assert diff.regressed
    assert diff.regressions[0].detail == "0.91->0.62"


def test_small_score_drop_within_tolerance_is_not_a_regression() -> None:
    base = _baseline(_scenario("s", _check("judge", True, 0.91)))
    now = _baseline(_scenario("s", _check("judge", True, 0.89)))
    assert not diff_baselines(base, now, score_tolerance=0.05).regressed


def test_improvement_is_not_a_regression() -> None:
    base = _baseline(_scenario("s", _check("judge", False, 0.40)))
    now = _baseline(_scenario("s", _check("judge", True, 0.85)))
    diff = diff_baselines(base, now)
    assert not diff.regressed
    assert diff.improvements[0].detail == "0.40->0.85"


def test_new_check_is_reported_but_does_not_gate() -> None:
    base = _baseline(_scenario("s", _check("a", True)))
    now = _baseline(_scenario("s", _check("a", True), _check("b", False)))
    diff = diff_baselines(base, now)
    assert not diff.regressed
    assert [d.check for d in diff.new_checks] == ["b"]


# --- blocking vs advisory levels ---------------------------------------------


def test_advisory_regression_is_reported_but_does_not_gate() -> None:
    base = _baseline(_scenario("s", _advisory("criterion:tone", True)))
    now = _baseline(_scenario("s", _advisory("criterion:tone", False)))
    diff = diff_baselines(base, now)
    assert not diff.regressed  # an advisory flip never blocks the merge
    assert not diff.blocking_regressions
    assert [d.check for d in diff.advisory_regressions] == ["criterion:tone"]
    assert "no regressions across 1 scenario(s)" in diff.summary()
    assert "1 advisory warning(s): criterion:tone pass->fail" in diff.summary()


def test_blocking_and_advisory_regressions_are_distinguished() -> None:
    base = _baseline(
        _scenario(
            "s", _check("called_tool('x')", True), _advisory("criterion:tone", True)
        )
    )
    now = _baseline(
        _scenario(
            "s", _check("called_tool('x')", False), _advisory("criterion:tone", False)
        )
    )
    diff = diff_baselines(base, now)
    assert diff.regressed  # the blocking one gates
    assert [d.check for d in diff.blocking_regressions] == ["called_tool('x')"]
    assert [d.check for d in diff.advisory_regressions] == ["criterion:tone"]
    # The gate headline counts only the blocking regression; the advisory tails it.
    assert diff.summary().startswith(
        "1/1 scenarios regressed: called_tool('x') pass->fail"
    )
    assert "1 advisory warning(s)" in diff.summary()


def test_summary_counts_regressed_scenarios() -> None:
    base = _baseline(
        _scenario("one", _check("c", True)),
        _scenario("two", _check("c", True)),
    )
    now = _baseline(
        _scenario("one", _check("c", False)),
        _scenario("two", _check("c", True)),
    )
    diff = diff_baselines(base, now)
    assert diff.summary().startswith("1/2 scenarios regressed:")
    assert diff.regressed_scenarios == ["one"]


# --- the ci CLI verb ---------------------------------------------------------


def _weather_cassette(
    path: Path, *, answer: str = "It is 18C and cloudy in Paris."
) -> None:
    question = [{"role": "user", "content": "What's the weather in Paris?"}]
    request2: dict[str, Any] = {"messages": question, "step": 2}
    save_cassette(
        Trajectory(
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
                    request_hash=hash_request(redact({"messages": question})),
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
                    request=request2,
                    response={"content": [{"type": "text", "text": answer}]},
                    request_hash=hash_request(redact(request2)),
                ),
            ],
        ),
        path,
    )


def _write_suite(directory: Path) -> Path:
    suite = directory / "suite.toml"
    suite.write_text(
        """
[[scenario]]
name = "weather"
cassette = "weather.json"

  [[scenario.check]]
  type = "called_tool"
  name = "get_weather"

  [[scenario.check]]
  type = "answer_contains"
  value = "cloudy"
""",
        encoding="utf-8",
    )
    return suite


def test_write_baseline_then_ci_is_clean(tmp_path: Path) -> None:
    _weather_cassette(tmp_path / "weather.json")
    suite = _write_suite(tmp_path)
    baseline = tmp_path / "baseline.json"
    written = runner.invoke(
        app, ["eval", str(suite), "--write-baseline", str(baseline)]
    )
    assert written.exit_code == 0, written.stdout
    assert baseline.is_file()

    result = runner.invoke(app, ["ci", str(suite), "--baseline", str(baseline)])
    assert result.exit_code == 0, result.stdout
    assert "no regressions" in result.stdout


def test_ci_detects_a_seeded_regression(tmp_path: Path) -> None:
    cassette = tmp_path / "weather.json"
    _weather_cassette(cassette)
    suite = _write_suite(tmp_path)
    baseline = tmp_path / "baseline.json"
    runner.invoke(app, ["eval", str(suite), "--write-baseline", str(baseline)])

    # Seed a regression: the agent no longer mentions the recorded conditions.
    _weather_cassette(cassette, answer="It is sunny in Paris.")
    result = runner.invoke(app, ["ci", str(suite), "--baseline", str(baseline)])
    assert result.exit_code == 1
    assert "scenarios regressed" in result.stdout
    assert "pass->fail" in result.stdout


def test_ci_writes_diff_json(tmp_path: Path) -> None:
    cassette = tmp_path / "weather.json"
    _weather_cassette(cassette)
    suite = _write_suite(tmp_path)
    baseline = tmp_path / "baseline.json"
    runner.invoke(app, ["eval", str(suite), "--write-baseline", str(baseline)])

    # Seed a regression and capture the diff JSON; it must be written even on exit 1.
    _weather_cassette(cassette, answer="It is sunny in Paris.")
    diff_json = tmp_path / "diff.json"
    result = runner.invoke(
        app,
        ["ci", str(suite), "--baseline", str(baseline), "--json", str(diff_json)],
    )
    assert result.exit_code == 1
    restored = RegressionDiff.model_validate_json(diff_json.read_text(encoding="utf-8"))
    assert restored.regressed
    assert restored.regressed_scenarios == ["weather"]
    # The recorded task rides into the JSON, so the Action can show it in the comment.
    assert restored.scenario_tasks["weather"] == "What's the weather in Paris?"


def test_ci_missing_baseline_errors(tmp_path: Path) -> None:
    _weather_cassette(tmp_path / "weather.json")
    suite = _write_suite(tmp_path)
    result = runner.invoke(
        app, ["ci", str(suite), "--baseline", str(tmp_path / "absent.json")]
    )
    assert result.exit_code == 2
