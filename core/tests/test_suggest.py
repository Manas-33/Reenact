"""`reenact suggest`: the structural deriver, keyword heuristic, and rendering.

The deriver is deterministic and keyless, so these run offline. Synthetic
trajectories isolate each rule; the real demo baseline cassettes pin the keyword
heuristic to actual recordings (real-first), and a round-trip proves the emitted
TOML loads back through ``load_suite`` to the checks it described.
"""

from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from reenact.cli import app
from reenact.evals import (
    CheckSuggestion,
    Criterion,
    ScenarioSuggestion,
    load_suite,
    render_suite_toml,
    run_suite,
    suggest_criteria,
    suggest_structural,
)
from reenact.schema import LLMCallEvent, SideEffect, ToolCallEvent, Trajectory
from reenact.store import load_cassette, save_cassette

runner = CliRunner()
DEMO_BASELINE = (
    Path(__file__).resolve().parent.parent
    / "examples"
    / "demo"
    / "scenarios"
    / "baseline"
)


def _traj(
    prompt: str,
    answer: str,
    tools: list[tuple[str, SideEffect]],
) -> Trajectory:
    """A think -> act* -> think recording: prompt in the first request, answer last."""
    events: list[Any] = [
        LLMCallEvent(
            seq=0,
            provider="p",
            model="m",
            request={"messages": [{"role": "user", "content": prompt}]},
            response={"content": []},
            request_hash="sha256:req0",
        )
    ]
    for index, (name, side_effect) in enumerate(tools, start=1):
        events.append(ToolCallEvent(seq=index, name=name, side_effect=side_effect))
    events.append(
        LLMCallEvent(
            seq=len(tools) + 1,
            provider="p",
            model="m",
            request={"messages": [{"role": "user", "content": "continue"}]},
            response={"content": [{"type": "text", "text": answer}]},
            request_hash="sha256:reqN",
        )
    )
    return Trajectory(name="t", events=events)


def _types(suggestions: list[CheckSuggestion]) -> list[str]:
    return [s.type for s in suggestions]


def _named(suggestions: list[CheckSuggestion], type_: str) -> list[CheckSuggestion]:
    return [s for s in suggestions if s.type == type_]


def _render_one(
    name: str,
    cassette: str,
    checks: list[CheckSuggestion],
    criteria: list[Criterion] | None = None,
) -> str:
    """Render a single-scenario suite (the common case in these tests)."""
    return render_suite_toml(
        [
            ScenarioSuggestion(
                name=name, cassette=cassette, checks=checks, criteria=criteria or []
            )
        ]
    )


class _StubResponse:
    def __init__(self, text: str) -> None:
        self._text = text

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        return {"content": [{"type": "text", "text": self._text}]}


class _RecordingMessages:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    def create(self, **_kwargs: Any) -> _StubResponse:
        self.calls += 1
        return _StubResponse(self._text)


class _StubProposer:
    """A duck-typed model client that returns a canned proposal and counts calls."""

    def __init__(self, text: str) -> None:
        self.messages = _RecordingMessages(text)


class _RaisingMessages:
    def create(self, **_kwargs: Any) -> Any:
        raise RuntimeError("boom")


class _RaisingProposer:
    def __init__(self) -> None:
        self.messages = _RaisingMessages()


# --- structural rules --------------------------------------------------------


def test_called_tool_per_distinct_tool_in_order() -> None:
    traj = _traj(
        "hello",
        "done",
        [
            ("search_docs", SideEffect.READ_ONLY),
            ("search_docs", SideEffect.READ_ONLY),
            ("label_issue", SideEffect.MUTATING),
        ],
    )
    called = _named(suggest_structural(traj), "called_tool")
    assert [s.params["name"] for s in called] == ["search_docs", "label_issue"]
    assert all(s.active for s in called)


def test_safety_check_when_a_mutating_tool_is_present() -> None:
    traj = _traj("hi", "bye", [("post_reply", SideEffect.MUTATING)])
    assert "no_mutating_tool_reexecuted" in _types(suggest_structural(traj))


def test_safety_check_when_a_tool_is_unknown() -> None:
    # UNKNOWN is treated as mutating on replay, so the safety net still applies.
    traj = _traj("hi", "bye", [("mystery", SideEffect.UNKNOWN)])
    assert "no_mutating_tool_reexecuted" in _types(suggest_structural(traj))


def test_no_safety_check_when_all_tools_read_only() -> None:
    traj = _traj("hi", "bye", [("search_docs", SideEffect.READ_ONLY)])
    assert "no_mutating_tool_reexecuted" not in _types(suggest_structural(traj))


def test_tool_call_count_offered_commented() -> None:
    traj = _traj("hi", "bye", [("search_docs", SideEffect.READ_ONLY)])
    counts = _named(suggest_structural(traj), "tool_call_count")
    assert len(counts) == 1
    assert counts[0].active is False
    assert counts[0].params == {"name": "search_docs", "count": 1}


def test_no_tools_yields_no_tool_checks() -> None:
    traj = _traj("just a question", "just an answer", [])
    types = _types(suggest_structural(traj))
    assert "called_tool" not in types
    assert "no_mutating_tool_reexecuted" not in types
    assert "tool_call_count" not in types


# --- keyword heuristic -------------------------------------------------------


def _keyword(traj: Trajectory) -> str | None:
    matches = _named(suggest_structural(traj), "answer_contains")
    return str(matches[0].params["value"]) if matches else None


def test_keyword_is_a_topical_anchor() -> None:
    # "password" and "reset" appear in both; the longest wins.
    traj = _traj(
        "Password reset link doesn't work",
        "To reset your password, request a new link.",
        [],
    )
    assert _keyword(traj) == "password"


def test_keyword_prefers_a_distinctive_code_over_a_longer_word() -> None:
    # 429 (digit) beats the longer shared word "checkout".
    traj = _traj(
        "Error 429 on checkout",
        "The 429 response happened during checkout.",
        [],
    )
    assert _keyword(traj) == "429"


def test_keyword_abstains_without_overlap() -> None:
    # The agent's category ("billing") is not a word the user typed - abstain.
    traj = _traj(
        "Charged twice this month",
        "Labeled as billing; duplicate charges are refunded in five days.",
        [],
    )
    assert _keyword(traj) is None


def test_keyword_skips_stopwords_and_short_words() -> None:
    traj = _traj("Is it up?", "It is up now.", [])
    assert _keyword(traj) is None


# --- pinned to the real demo recordings (real-first) -------------------------


def test_real_baseline_cassettes_match_expected_keywords() -> None:
    expected = {
        "issue-42.json": "password",
        "issue-57.json": None,  # "billing" is answer-only; the deriver abstains
        "issue-63.json": "429",
    }
    for filename, keyword in expected.items():
        traj = load_cassette(DEMO_BASELINE / filename)
        suggestions = suggest_structural(traj)
        assert _keyword(traj) == keyword, filename
        # Every real run used the three triage tools and a mutating one.
        called = {str(s.params["name"]) for s in _named(suggestions, "called_tool")}
        assert {"search_docs", "label_issue", "post_reply"} <= called, filename
        assert "no_mutating_tool_reexecuted" in _types(suggestions), filename


# --- rendering + round-trip --------------------------------------------------


def test_render_round_trips_through_load_suite(tmp_path: Path) -> None:
    cassette = tmp_path / "run.json"
    traj = _traj(
        "Password reset link is broken",
        "To reset your password, open a new link.",
        [
            ("search_docs", SideEffect.READ_ONLY),
            ("post_reply", SideEffect.MUTATING),
        ],
    )
    save_cassette(traj, cassette)
    body = _render_one("run", str(cassette), suggest_structural(traj))
    suite = tmp_path / "suite.toml"
    suite.write_text(body, encoding="utf-8")

    report = run_suite(load_suite(suite))
    names = [c.name for c in report.scenarios[0].checks]
    # Only the active tables load; the commented tool_call_count does not.
    assert names == [
        "called_tool('search_docs')",
        "called_tool('post_reply')",
        "no_mutating_tool_reexecuted",
        "answer_contains('password')",
    ]
    assert report.passed


def test_render_marks_commented_and_criterion_lines() -> None:
    traj = _traj("hi", "bye", [("search_docs", SideEffect.READ_ONLY)])
    body = _render_one("t", "run.json", suggest_structural(traj))
    # The tighter alternative and the criterion placeholder are commented out.
    assert '  # type = "tool_call_count"' in body
    assert "#   [[scenario.criterion]]" in body
    assert "keep what applies" in body


def test_criterion_placeholder_is_always_commented() -> None:
    # An active criterion would need a client at load time; ours must stay inert.
    traj = _traj("hi", "bye", [("search_docs", SideEffect.READ_ONLY)])
    body = _render_one("t", "run.json", suggest_structural(traj))
    criterion_lines = [ln for ln in body.splitlines() if "scenario.criterion" in ln]
    assert criterion_lines
    assert all(ln.lstrip().startswith("#") for ln in criterion_lines)


# --- CLI ---------------------------------------------------------------------


def test_suggest_prints_to_stdout(tmp_path: Path) -> None:
    cassette = tmp_path / "run.json"
    save_cassette(_traj("hi", "bye", [("search_docs", SideEffect.READ_ONLY)]), cassette)
    result = runner.invoke(app, ["suggest", str(cassette)])
    assert result.exit_code == 0, result.stdout
    assert "[[scenario]]" in result.stdout
    assert 'type = "called_tool"' in result.stdout


def test_suggest_writes_with_output_flag(tmp_path: Path) -> None:
    cassette = tmp_path / "run.json"
    traj = _traj(
        "Password reset is broken",
        "Reset your password via a new link.",
        [("search_docs", SideEffect.READ_ONLY), ("post_reply", SideEffect.MUTATING)],
    )
    save_cassette(traj, cassette)
    out = tmp_path / "suite.toml"
    result = runner.invoke(app, ["suggest", str(cassette), "-o", str(out)])
    assert result.exit_code == 0, result.stdout
    assert out.is_file()
    # What it wrote is a runnable suite.
    report = run_suite(load_suite(out))
    assert report.passed


def test_suggest_output_path_resolves_from_a_subdir(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # The init dovetail: `suggest evals/scenarios/run.json -o evals/suite.toml` run
    # from the project root must write a cassette path load_suite resolves relative to
    # the suite's own dir (i.e. "scenarios/run.json", not "evals/scenarios/run.json").
    monkeypatch.chdir(tmp_path)
    scenarios = tmp_path / "evals" / "scenarios"
    scenarios.mkdir(parents=True)
    save_cassette(
        _traj(
            "Password reset is broken",
            "Reset your password via a new link.",
            [
                ("search_docs", SideEffect.READ_ONLY),
                ("post_reply", SideEffect.MUTATING),
            ],
        ),
        scenarios / "run.json",
    )
    result = runner.invoke(
        app, ["suggest", "evals/scenarios/run.json", "-o", "evals/suite.toml"]
    )
    assert result.exit_code == 0, result.stdout
    suite = tmp_path / "evals" / "suite.toml"
    assert 'cassette = "scenarios/run.json"' in suite.read_text(encoding="utf-8")
    assert run_suite(load_suite(suite)).passed  # resolves + runs from the suite dir


def test_suggest_multiple_cassettes_into_one_suite(tmp_path: Path) -> None:
    # Several cassettes -> one suite with a [[scenario]] each.
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    save_cassette(_traj("hi a", "bye a", [("t", SideEffect.READ_ONLY)]), a)
    save_cassette(_traj("hi b", "bye b", [("t", SideEffect.READ_ONLY)]), b)
    result = runner.invoke(app, ["suggest", str(a), str(b)])
    assert result.exit_code == 0, result.stdout
    assert result.stdout.count("[[scenario]]") == 2
    assert str(a) in result.stdout and str(b) in result.stdout


def test_suggest_expands_a_directory_of_cassettes(tmp_path: Path) -> None:
    # A directory argument expands to its *.json cassettes.
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    for name in ("one", "two", "three"):
        save_cassette(
            _traj("hi", "bye", [("t", SideEffect.READ_ONLY)]),
            scenarios / f"{name}.json",
        )
    result = runner.invoke(app, ["suggest", str(scenarios)])
    assert result.exit_code == 0, result.stdout
    assert result.stdout.count("[[scenario]]") == 3


# --- B2: optional AI quality-criteria layer ----------------------------------

CRITERIA_JSON = (
    '[{"id": "reply_grounded", "question": "Is the reply grounded in the docs?"},'
    ' {"id": "correct_label", "question": "Is the applied label appropriate?"}]'
)


def test_suggest_criteria_parses_and_calls_once() -> None:
    stub = _StubProposer(CRITERIA_JSON)
    traj = _traj("hi", "bye", [("search_docs", SideEffect.READ_ONLY)])
    criteria = suggest_criteria(stub, traj)
    assert [c.id for c in criteria] == ["reply_grounded", "correct_label"]
    assert stub.messages.calls == 1  # a single batched call


def test_suggest_criteria_ignores_a_garbled_reply() -> None:
    traj = _traj("hi", "bye", [])
    assert suggest_criteria(_StubProposer("sorry, no JSON here"), traj) == []
    assert suggest_criteria(_StubProposer("[]"), traj) == []


def test_suggest_criteria_drops_invalid_and_duplicate_items() -> None:
    reply = (
        '[{"id": "ok", "question": "Good?"},'
        ' {"id": "ok", "question": "duplicate id"},'  # dropped: duplicate id
        ' {"id": "missing_q"},'  # dropped: no question
        ' {"question": "no id"},'  # dropped: no id
        ' "not an object"]'  # dropped: not a table
    )
    criteria = suggest_criteria(_StubProposer(reply), _traj("hi", "bye", []))
    assert [c.id for c in criteria] == ["ok"]


def test_render_includes_commented_criteria_and_round_trips(tmp_path: Path) -> None:
    cassette = tmp_path / "run.json"
    traj = _traj(
        "Password reset is broken",
        "Reset your password via a new link.",
        [("search_docs", SideEffect.READ_ONLY), ("post_reply", SideEffect.MUTATING)],
    )
    save_cassette(traj, cassette)
    criteria = [Criterion(id="reply_grounded", question="Grounded in the docs?")]
    body = _render_one("run", str(cassette), suggest_structural(traj), criteria)
    assert "proposed from the transcript" in body
    assert "reply_grounded" in body
    # Every criterion line is commented, so the suite still loads with no client.
    criterion_lines = [ln for ln in body.splitlines() if "scenario.criterion" in ln]
    assert criterion_lines and all(
        ln.lstrip().startswith("#") for ln in criterion_lines
    )
    suite = tmp_path / "suite.toml"
    suite.write_text(body, encoding="utf-8")
    assert run_suite(load_suite(suite)).passed


def test_suggest_cli_proposes_criteria_with_a_client(
    tmp_path: Path, monkeypatch: Any
) -> None:
    cassette = tmp_path / "run.json"
    save_cassette(_traj("hi", "bye", [("search_docs", SideEffect.READ_ONLY)]), cassette)
    stub = _StubProposer('[{"id": "grounded", "question": "Grounded?"}]')
    monkeypatch.setattr("reenact.cli._judge_client", lambda: stub)
    result = runner.invoke(app, ["suggest", str(cassette)])
    assert result.exit_code == 0, result.stdout
    assert "proposed from the transcript" in result.stdout
    assert "grounded" in result.stdout
    assert stub.messages.calls == 1


def test_suggest_cli_without_a_client_is_structural_only(
    tmp_path: Path, monkeypatch: Any
) -> None:
    cassette = tmp_path / "run.json"
    save_cassette(_traj("hi", "bye", [("search_docs", SideEffect.READ_ONLY)]), cassette)
    monkeypatch.setattr("reenact.cli._judge_client", lambda: None)
    result = runner.invoke(app, ["suggest", str(cassette)])
    assert result.exit_code == 0, result.stdout
    assert 'type = "called_tool"' in result.stdout
    # The illustrative example is shown, not real proposals.
    assert "sit beside" in result.stdout


def test_suggest_cli_no_ai_flag_skips_the_client(
    tmp_path: Path, monkeypatch: Any
) -> None:
    cassette = tmp_path / "run.json"
    save_cassette(_traj("hi", "bye", [("search_docs", SideEffect.READ_ONLY)]), cassette)
    stub = _StubProposer('[{"id": "grounded", "question": "Grounded?"}]')
    monkeypatch.setattr("reenact.cli._judge_client", lambda: stub)
    result = runner.invoke(app, ["suggest", str(cassette), "--no-ai"])
    assert result.exit_code == 0, result.stdout
    assert stub.messages.calls == 0  # the client is never consulted
    assert "proposed from the transcript" not in result.stdout  # no AI proposals


def test_suggest_cli_survives_a_client_error(tmp_path: Path, monkeypatch: Any) -> None:
    cassette = tmp_path / "run.json"
    save_cassette(_traj("hi", "bye", [("search_docs", SideEffect.READ_ONLY)]), cassette)
    monkeypatch.setattr("reenact.cli._judge_client", lambda: _RaisingProposer())
    result = runner.invoke(app, ["suggest", str(cassette)])
    assert result.exit_code == 0, result.stdout
    assert 'type = "called_tool"' in result.stdout  # structural still emitted
