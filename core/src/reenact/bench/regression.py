"""Regression bench: seeded-regression catch rate paired with false-positive rate.

Diffs a committed baseline against each recorded break-me set and the benign sets.
Catch rate is the fraction of seeded-regression branches the gate blocks; FPR is
the fraction of benign scenarios it falsely flags. The two are reported together
on purpose - a catch rate alone is meaningless, since a gate that flags everything
scores 100% catch and is useless (§8's survival metric: flaky gates get
uninstalled).

The demo baselines are the *structural* assertions (tool calls, mutating-tool
safety, a topical keyword) - deterministic and key-free, so the numbers reproduce
in CI. The trajectory judge is a supplementary quality lens (unit-tested
separately), not part of these headline numbers: on ambiguous tasks its
run-to-run noise is too large to gate on without flaking, which is itself the
point of the FPR floor.
"""

from pathlib import Path
from typing import Any

from reenact.evals import Baseline, diff_baselines, load_baseline

BASELINES = Path(__file__).resolve().parents[3] / "examples" / "demo" / "baselines"
CATCH_FLOOR = 0.90
FPR_FLOOR = 0.05
# Applies only to scored (judge) baselines; assertion baselines carry no score.
# Set above the LLM judge's ~0.15 run-to-run noise so a judged gate flags real
# regressions, not jitter.
JUDGE_TOLERANCE = 0.25

# Ground truth, verified by inspection at record time: which recorded sets injected
# a real regression, and which are benign changes the gate must not flag. A model
# swap to a capably cheaper model triaged identically here, so it is benign - the
# gate's specificity, not a miss.
_REGRESSION_SETS = ("prompt-edit", "tool-schema")
_BENIGN_SETS = ("model-swap", "clean-pr")


def measure_catch_fpr(
    baseline: Baseline,
    regressions: dict[str, Baseline],
    benign: dict[str, Baseline],
    *,
    tolerance: float = JUDGE_TOLERANCE,
) -> dict[str, Any]:
    """Catch rate over the regression sets, FPR over the benign sets' scenarios."""
    branches: list[dict[str, Any]] = []
    caught = 0
    for name, variant in regressions.items():
        diff = diff_baselines(baseline, variant, score_tolerance=tolerance)
        caught += 1 if diff.regressed else 0
        branches.append(
            {
                "set": name,
                "kind": "regression",
                "flagged": diff.regressed,
                "detail": diff.summary(),
            }
        )

    false_positive_scenarios = 0
    benign_scenarios = 0
    for name, variant in benign.items():
        diff = diff_baselines(baseline, variant, score_tolerance=tolerance)
        benign_scenarios += len(variant.scenarios)
        false_positive_scenarios += len(diff.regressed_scenarios)
        branches.append(
            {
                "set": name,
                "kind": "benign",
                "flagged": diff.regressed,
                "false_positive_scenarios": len(diff.regressed_scenarios),
            }
        )

    n_regressions = len(regressions)
    catch_rate = round(caught / n_regressions, 4) if n_regressions else 0.0
    fpr = (
        round(false_positive_scenarios / benign_scenarios, 4)
        if benign_scenarios
        else 0.0
    )
    return {
        "metric": "regression_catch_and_fpr",
        "catch_rate": catch_rate,
        "caught": caught,
        "regressions_total": n_regressions,
        "catch_floor": CATCH_FLOOR,
        "catch_within_floor": catch_rate >= CATCH_FLOOR,
        "fpr": fpr,
        "false_positive_scenarios": false_positive_scenarios,
        "benign_scenarios_total": benign_scenarios,
        "fpr_floor": FPR_FLOOR,
        "fpr_within_floor": fpr <= FPR_FLOOR,
        "branches": branches,
    }


def measure_demo_regression(baselines: Path = BASELINES) -> dict[str, Any]:
    """Load the committed demo baselines and measure catch rate + FPR."""
    baseline_path = baselines / "baseline.json"
    if not baseline_path.is_file():
        return {"metric": "regression_catch_and_fpr", "regressions_total": 0}
    baseline = load_baseline(baseline_path)
    regressions = {
        name: load_baseline(baselines / f"{name}.json") for name in _REGRESSION_SETS
    }
    benign = {name: load_baseline(baselines / f"{name}.json") for name in _BENIGN_SETS}
    return measure_catch_fpr(baseline, regressions, benign)
