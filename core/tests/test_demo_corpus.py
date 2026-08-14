"""The recorded demo corpus: the gate catches real regressions, passes clean PRs.

These replay the committed cassettes (recorded once against the live model) with
zero network. They prove the whole wedge on real data: the baseline suite passes,
a benign PR does not regress, a tool-schema change is caught on every scenario,
and a model swap slips past *assertions* alone - which is exactly why the judge
(the next rung) exists.
"""

from pathlib import Path

from demo.suites import demo_scenarios

from reenact.evals import Baseline, diff_baselines, load_baseline, run_suite

DEMO = Path(__file__).resolve().parent.parent / "examples" / "demo"


def _baseline_of(set_name: str) -> Baseline:
    return Baseline.from_report(run_suite(demo_scenarios(set_name)))


def test_baseline_suite_passes_offline() -> None:
    report = run_suite(demo_scenarios("baseline"))
    assert report.total == 3
    assert report.passed, [c.message for s in report.scenarios for c in s.failures]


def test_committed_baseline_matches_the_recordings() -> None:
    committed = load_baseline(DEMO / "baseline.json")
    assert not diff_baselines(committed, _baseline_of("baseline")).regressed


def test_clean_pr_does_not_regress() -> None:
    diff = diff_baselines(_baseline_of("baseline"), _baseline_of("clean-pr"))
    assert not diff.regressed, diff.summary()


def test_tool_schema_change_is_caught() -> None:
    diff = diff_baselines(_baseline_of("baseline"), _baseline_of("tool-schema"))
    assert diff.regressed
    assert len(diff.regressed_scenarios) == 3  # every scenario loses label_issue
    assert all("label_issue" in delta.check for delta in diff.regressions)


def test_prompt_edit_is_caught() -> None:
    diff = diff_baselines(_baseline_of("baseline"), _baseline_of("prompt-edit"))
    assert diff.regressed


def test_model_swap_slips_past_assertions() -> None:
    # A model swap keeps the same tool calls, so structural assertions see no
    # regression - only the trajectory judge (a later rung) catches the quality
    # drop. This documents why the judge is needed, not a bug.
    diff = diff_baselines(_baseline_of("baseline"), _baseline_of("model-swap"))
    assert not diff.regressed
