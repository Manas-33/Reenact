"""The regression bench: catch rate paired with false-positive rate.

The diff logic is tested on synthetic baselines (each case isolates one behavior);
the last test runs the real committed demo baselines and asserts both §8 floors -
catch rate >= 90%, FPR <= 5% - so the numbers can never silently drop below floor.
"""

from reenact.bench.regression import (
    CATCH_FLOOR,
    FPR_FLOOR,
    measure_catch_fpr,
    measure_demo_regression,
)
from reenact.evals import Baseline
from reenact.evals.baseline import BaselineCheck, BaselineScenario


def _c(name: str, passed: bool, score: float | None = None) -> BaselineCheck:
    return BaselineCheck(name=name, passed=passed, score=score)


def _s(name: str, *checks: BaselineCheck) -> BaselineScenario:
    return BaselineScenario(name=name, checks=list(checks))


def _bl(*scenarios: BaselineScenario) -> Baseline:
    return Baseline(scenarios=list(scenarios))


def test_catch_and_fpr_on_a_fixture() -> None:
    baseline = _bl(_s("s", _c("called_tool('x')", True), _c("judge", True, 0.9)))
    regressed = _bl(_s("s", _c("called_tool('x')", False), _c("judge", True, 0.9)))
    benign = _bl(_s("s", _c("called_tool('x')", True), _c("judge", True, 0.88)))
    metrics = measure_catch_fpr(baseline, {"bad": regressed}, {"ok": benign})
    assert metrics["catch_rate"] == 1.0
    assert metrics["catch_within_floor"]
    assert metrics["fpr"] == 0.0
    assert metrics["fpr_within_floor"]


def test_tolerance_absorbs_judge_noise() -> None:
    baseline = _bl(_s("s", _c("judge", True, 0.9)))
    noisy = _bl(_s("s", _c("judge", True, 0.75)))  # a 0.15 run-to-run wobble
    tight = measure_catch_fpr(baseline, {}, {"ok": noisy}, tolerance=0.05)
    assert tight["fpr"] == 1.0  # tight tolerance false-positives on judge noise
    wide = measure_catch_fpr(baseline, {}, {"ok": noisy}, tolerance=0.25)
    assert wide["fpr"] == 0.0  # noise-aware tolerance absorbs it


def test_missed_regression_drops_below_floor() -> None:
    baseline = _bl(_s("s", _c("judge", True, 0.9)))
    unchanged = _bl(_s("s", _c("judge", True, 0.9)))  # a regression the gate can't see
    metrics = measure_catch_fpr(baseline, {"missed": unchanged}, {})
    assert metrics["catch_rate"] == 0.0
    assert not metrics["catch_within_floor"]


def test_demo_regression_meets_floors() -> None:
    metrics = measure_demo_regression()
    assert metrics["regressions_total"] == 2  # tool-schema, prompt-edit
    assert metrics["benign_scenarios_total"] == 6  # model-swap + clean-pr, 3 each
    assert metrics["catch_rate"] >= CATCH_FLOOR, metrics["branches"]
    assert metrics["catch_within_floor"], metrics["branches"]
    assert metrics["fpr"] <= FPR_FLOOR, metrics["branches"]
    assert metrics["fpr_within_floor"], metrics["branches"]
