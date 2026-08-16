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
from reenact.evals.calibration import (
    CalibrationReport,
    CriterionAgreement,
    Label,
    LabelSet,
    calibrate,
    load_label_set,
    save_label_set,
)
from reenact.evals.check import (
    Check,
    CheckResult,
    CriterionLevel,
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
    PairwiseVerdict,
    StructuredEvaluator,
    pairwise,
    structured_eval,
)
from reenact.evals.suggest import (
    CheckSuggestion,
    render_suite_toml,
    suggest_structural,
)
from reenact.evals.suite import SuiteConfigError, load_suite

__all__ = [
    "FAITHFULNESS",
    "Baseline",
    "CalibrationReport",
    "Check",
    "CheckDelta",
    "CheckResult",
    "CheckSuggestion",
    "Criterion",
    "CriterionAgreement",
    "CriterionLevel",
    "CriterionVerdict",
    "DeltaKind",
    "EvalReport",
    "JudgeVerdict",
    "Label",
    "LabelSet",
    "PairwiseVerdict",
    "RegressionDiff",
    "RunView",
    "Scenario",
    "ScenarioResult",
    "StructuredEvaluator",
    "SuiteConfigError",
    "answer_contains",
    "answer_matches",
    "calibrate",
    "called_tool",
    "did_not_call_tool",
    "diff_baselines",
    "judged",
    "load_baseline",
    "load_label_set",
    "load_suite",
    "no_mutating_tool_reexecuted",
    "pairwise",
    "render_suite_toml",
    "render_trajectory",
    "replays_clean",
    "run_scenario",
    "run_suite",
    "save_baseline",
    "save_label_set",
    "structured_eval",
    "suggest_structural",
    "tool_call_count",
]
