"""Evals: the scenario runner (a recording plus checks), plain-Python
assertions, and a trajectory-level LLM judge that scores multi-step behavior.
"""

from reenact.evals.baseline import (
    Baseline,
    CheckDelta,
    DeltaKind,
    RegressionDiff,
    diff_baselines,
    load_baseline,
    save_baseline,
)
from reenact.evals.check import (
    Check,
    CheckResult,
    RunView,
    answer_contains,
    answer_matches,
    called_tool,
    did_not_call_tool,
    no_mutating_tool_reexecuted,
    replays_clean,
    tool_call_count,
)
from reenact.evals.judge import JudgeVerdict, judged, render_trajectory
from reenact.evals.runner import (
    EvalReport,
    ScenarioResult,
    run_scenario,
    run_suite,
)
from reenact.evals.scenario import Scenario
from reenact.evals.structured import (
    FAITHFULNESS,
    Criterion,
    CriterionVerdict,
    StructuredEvaluator,
    structured_eval,
)
from reenact.evals.suite import SuiteConfigError, load_suite

__all__ = [
    "FAITHFULNESS",
    "Baseline",
    "Check",
    "CheckDelta",
    "CheckResult",
    "Criterion",
    "CriterionVerdict",
    "DeltaKind",
    "EvalReport",
    "JudgeVerdict",
    "RegressionDiff",
    "RunView",
    "Scenario",
    "ScenarioResult",
    "StructuredEvaluator",
    "SuiteConfigError",
    "answer_contains",
    "answer_matches",
    "called_tool",
    "did_not_call_tool",
    "diff_baselines",
    "judged",
    "load_baseline",
    "load_suite",
    "no_mutating_tool_reexecuted",
    "render_trajectory",
    "replays_clean",
    "run_scenario",
    "run_suite",
    "save_baseline",
    "structured_eval",
    "tool_call_count",
]
