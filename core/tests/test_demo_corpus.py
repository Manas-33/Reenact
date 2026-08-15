"""The recorded demo corpus: the gate catches real regressions, passes clean PRs.

These replay the committed cassettes (recorded once against the live model) with
zero network. They prove the whole wedge on real data: the baseline suite passes,
a tool-schema change and a degraded prompt are caught, and both a benign reword
and a capable model swap pass untouched - the gate's specificity, which is what
keeps the false-positive rate at zero.
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


def test_model_swap_is_benign() -> None:
    # A swap to a capably cheaper model triaged identically here (same tools, same
    # grounded answers), so it is genuinely not a regression - the gate correctly
    # passes it. Flagging every model change would be the flaky gate that gets
    # uninstalled; this is the specificity the FPR floor protects.
    diff = diff_baselines(_baseline_of("baseline"), _baseline_of("model-swap"))
    assert not diff.regressed
