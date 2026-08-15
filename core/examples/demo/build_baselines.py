"""Build the committed assertion baselines for the demo regression bench.

Runs the demo suite (assertions only - no judge, no key) over each recorded set
and writes a baseline per set to ``examples/demo/baselines/``, so the catch-rate /
FPR bench diffs them offline and deterministically. Regenerate after re-recording:

    PYTHONPATH=examples python examples/demo/build_baselines.py
"""

from pathlib import Path

from demo.suites import demo_scenarios
from reenact.evals import Baseline, run_suite, save_baseline

BASELINES = Path(__file__).resolve().parent / "baselines"
SETS = ["baseline", "model-swap", "prompt-edit", "tool-schema", "clean-pr"]


def main() -> None:
    BASELINES.mkdir(parents=True, exist_ok=True)
    for name in SETS:
        report = run_suite(demo_scenarios(name))
        save_baseline(Baseline.from_report(report), BASELINES / f"{name}.json")
        print(f"  {name}: {report.passed_count}/{report.total} scenarios pass")


if __name__ == "__main__":
    main()
