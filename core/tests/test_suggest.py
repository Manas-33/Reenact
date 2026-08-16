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
    load_suite,
    render_suite_toml,
    run_suite,
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
    body = render_suite_toml("run", str(cassette), suggest_structural(traj))
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
    body = render_suite_toml("t", "run.json", suggest_structural(traj))
    # The tighter alternative and the criterion placeholder are commented out.
    assert '  # type = "tool_call_count"' in body
    assert "#   [[scenario.criterion]]" in body
    assert "keep what applies" in body


def test_criterion_placeholder_is_always_commented() -> None:
    # An active criterion would need a client at load time; ours must stay inert.
    traj = _traj("hi", "bye", [("search_docs", SideEffect.READ_ONLY)])
    body = render_suite_toml("t", "run.json", suggest_structural(traj))
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
