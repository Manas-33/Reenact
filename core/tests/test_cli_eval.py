"""The `reenact eval` and `reenact record` CLI verbs, plus the suite loader.

The suites here run offline over synthetic cassettes (and one real corpus
cassette), so nothing hits the network. Judge wiring is proven through
``load_suite`` with a stub client - the live judge path needs a key and is not
exercised here.
"""

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from reenact.cli import app
from reenact.evals import CriterionLevel, SuiteConfigError, load_suite, run_suite
from reenact.record import hash_request, redact
from reenact.schema import LLMCallEvent, SideEffect, ToolCallEvent, Trajectory
from reenact.store import load_cassette, save_cassette

runner = CliRunner()
CORPUS = Path(__file__).resolve().parent.parent / "examples" / "corpus"


class _StubResponse:
    def __init__(self, text: str) -> None:
        self._text = text

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        return {"content": [{"type": "text", "text": self._text}]}


class _StubMessages:
    def __init__(self, text: str) -> None:
        self._text = text

    def create(self, **_kwargs: Any) -> _StubResponse:
        return _StubResponse(self._text)


class _StubJudgeClient:
    def __init__(self, text: str) -> None:
        self.messages = _StubMessages(text)


def _weather_cassette(path: Path) -> Path:
    question = [{"role": "user", "content": "What's the weather in Paris?"}]
    request2: dict[str, Any] = {"messages": question, "step": 2}
    trajectory = Trajectory(
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
                response={
                    "content": [
                        {"type": "text", "text": "It is 18C and cloudy in Paris."}
                    ]
                },
                request_hash=hash_request(redact(request2)),
            ),
        ],
    )
    save_cassette(trajectory, path)
    return path


# --- reenact eval ------------------------------------------------------------


def test_eval_reports_clean_suite(tmp_path: Path) -> None:
    _weather_cassette(tmp_path / "weather.json")
    suite = tmp_path / "suite.toml"
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

  [[scenario.check]]
  type = "replays_clean"
""",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["eval", str(suite)])
    assert result.exit_code == 0, result.stdout
    assert "PASS weather" in result.stdout
    assert "1/1 scenarios passed" in result.stdout


def test_eval_fails_on_failing_assertion(tmp_path: Path) -> None:
    _weather_cassette(tmp_path / "weather.json")
    suite = tmp_path / "suite.toml"
    suite.write_text(
        """
[[scenario]]
name = "weather"
cassette = "weather.json"

  [[scenario.check]]
  type = "answer_contains"
  value = "Berlin"
""",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["eval", str(suite)])
    assert result.exit_code == 1
    assert "FAIL weather" in result.stdout
    assert "answer_contains('Berlin')" in result.stdout


def test_eval_unknown_check_type_errors(tmp_path: Path) -> None:
    _weather_cassette(tmp_path / "weather.json")
    suite = tmp_path / "suite.toml"
    suite.write_text(
        '[[scenario]]\ncassette = "weather.json"\n[[scenario.check]]\ntype = "bogus"\n',
        encoding="utf-8",
    )
    result = runner.invoke(app, ["eval", str(suite)])
    assert result.exit_code == 2
    assert "unknown check type" in result.stdout


def test_eval_judge_without_client_errors(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _weather_cassette(tmp_path / "weather.json")
    suite = tmp_path / "suite.toml"
    suite.write_text(
        '[[scenario]]\ncassette = "weather.json"\n'
        '[[scenario.check]]\ntype = "judge"\nrubric = "reports the weather"\n',
        encoding="utf-8",
    )
    result = runner.invoke(app, ["eval", str(suite)])
    assert result.exit_code == 2
    assert "judge client" in result.stdout


def test_eval_over_real_corpus_cassette(tmp_path: Path) -> None:
    cassette = CORPUS / "tool_use-00.json"
    suite = tmp_path / "suite.toml"
    suite.write_text(
        f'[[scenario]]\ncassette = "{cassette}"\n'
        '[[scenario.check]]\ntype = "called_tool"\nname = "get_weather"\n'
        '[[scenario.check]]\ntype = "answer_contains"\nvalue = "cloudy"\n',
        encoding="utf-8",
    )
    result = runner.invoke(app, ["eval", str(suite)])
    assert result.exit_code == 0, result.stdout
    assert "1/1 scenarios passed" in result.stdout


# --- suite loader (judge wiring, relative paths) -----------------------------


def test_load_suite_wires_judge_client(tmp_path: Path) -> None:
    _weather_cassette(tmp_path / "weather.json")
    suite = tmp_path / "suite.toml"
    suite.write_text(
        '[[scenario]]\nname = "weather"\ncassette = "weather.json"\n'
        '[[scenario.check]]\ntype = "judge"\nrubric = "reports the weather"\n'
        "threshold = 0.5\n",
        encoding="utf-8",
    )
    stub = _StubJudgeClient('{"score": 0.9, "reasoning": "correct"}')
    scenarios = load_suite(suite, judge_client=stub)
    report = run_suite(scenarios)
    assert report.passed
    assert report.scenarios[0].checks[0].score == 0.9


# --- suite loader (structured criteria) --------------------------------------


def test_load_suite_wires_criteria(tmp_path: Path) -> None:
    _weather_cassette(tmp_path / "weather.json")
    suite = tmp_path / "suite.toml"
    suite.write_text(
        '[[scenario]]\nname = "weather"\ncassette = "weather.json"\n'
        '[[scenario.criterion]]\nid = "grounded"\n'
        'question = "Is the answer grounded?"\n'
        '[[scenario.criterion]]\nid = "tone"\nquestion = "Polite?"\n'
        'level = "advisory"\n',
        encoding="utf-8",
    )
    stub = _StubJudgeClient(
        '[{"id": "grounded", "passed": true, "evidence": "[2]"},'
        ' {"id": "tone", "passed": true, "evidence": "[2]"}]'
    )
    report = run_suite(load_suite(suite, judge_client=stub))
    assert report.passed
    checks = report.scenarios[0].checks
    assert [c.name for c in checks] == ["criterion:grounded", "criterion:tone"]
    # The level authored in config rides onto the result.
    levels = {c.name: c.level for c in checks}
    assert levels["criterion:grounded"] is CriterionLevel.BLOCKING
    assert levels["criterion:tone"] is CriterionLevel.ADVISORY


def test_load_suite_criterion_without_client_errors(tmp_path: Path) -> None:
    _weather_cassette(tmp_path / "weather.json")
    suite = tmp_path / "suite.toml"
    suite.write_text(
        '[[scenario]]\ncassette = "weather.json"\n'
        '[[scenario.criterion]]\nid = "x"\nquestion = "?"\n',
        encoding="utf-8",
    )
    with pytest.raises(SuiteConfigError, match="criterion"):
        load_suite(suite)


def test_load_suite_rejects_bad_criterion_level(tmp_path: Path) -> None:
    _weather_cassette(tmp_path / "weather.json")
    suite = tmp_path / "suite.toml"
    suite.write_text(
        '[[scenario]]\ncassette = "weather.json"\n'
        '[[scenario.criterion]]\nid = "x"\nquestion = "?"\nlevel = "loud"\n',
        encoding="utf-8",
    )
    stub = _StubJudgeClient("[]")
    with pytest.raises(SuiteConfigError, match=r"blocking.*advisory"):
        load_suite(suite, judge_client=stub)


def test_eval_runs_a_criterion(tmp_path: Path, monkeypatch: Any) -> None:
    _weather_cassette(tmp_path / "weather.json")
    suite = tmp_path / "suite.toml"
    suite.write_text(
        '[[scenario]]\nname = "weather"\ncassette = "weather.json"\n'
        '[[scenario.criterion]]\nid = "grounded"\nquestion = "Grounded?"\n',
        encoding="utf-8",
    )
    stub = _StubJudgeClient('[{"id": "grounded", "passed": true, "evidence": "[2]"}]')
    monkeypatch.setattr("reenact.cli._judge_client", lambda: stub)
    result = runner.invoke(app, ["eval", str(suite)])
    assert result.exit_code == 0, result.stdout
    assert "PASS weather" in result.stdout


def test_eval_criterion_without_client_errors(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _weather_cassette(tmp_path / "weather.json")
    suite = tmp_path / "suite.toml"
    suite.write_text(
        '[[scenario]]\ncassette = "weather.json"\n'
        '[[scenario.criterion]]\nid = "x"\nquestion = "?"\n',
        encoding="utf-8",
    )
    result = runner.invoke(app, ["eval", str(suite)])
    assert result.exit_code == 2
    assert "criterion" in result.stdout


# --- reenact record ----------------------------------------------------------


def test_record_writes_cassette_from_trajectory(tmp_path: Path) -> None:
    module = tmp_path / "scenario_mod.py"
    module.write_text(
        "from reenact.schema import Trajectory\n\n\n"
        "def make():\n    return Trajectory(name='made', events=[])\n",
        encoding="utf-8",
    )
    out = tmp_path / "made.json"
    result = runner.invoke(app, ["record", f"{module}:make", str(out)])
    assert result.exit_code == 0, result.stdout
    assert out.is_file()
    assert load_cassette(out).name == "made"


def test_record_accepts_a_recorder(tmp_path: Path) -> None:
    module = tmp_path / "rec_mod.py"
    module.write_text(
        "from reenact.record import Recorder\n"
        "from reenact.schema import SideEffect\n\n\n"
        "def make():\n"
        "    rec = Recorder(name='rec')\n"
        "    rec.record_tool_call(name='post_reply', arguments={'body': 'hi'},\n"
        "                         result='ok', side_effect=SideEffect.MUTATING)\n"
        "    return rec\n",
        encoding="utf-8",
    )
    out = tmp_path / "rec.json"
    result = runner.invoke(app, ["record", f"{module}:make", str(out)])
    assert result.exit_code == 0, result.stdout
    loaded = load_cassette(out)
    tool_names = [e.name for e in loaded.events if isinstance(e, ToolCallEvent)]
    assert tool_names == ["post_reply"]


def test_record_rejects_missing_entrypoint(tmp_path: Path) -> None:
    module = tmp_path / "scenario_mod.py"
    module.write_text("def make():\n    return None\n", encoding="utf-8")
    out = tmp_path / "out.json"
    result = runner.invoke(app, ["record", f"{module}:missing", str(out)])
    assert result.exit_code == 2
    assert not out.exists()
