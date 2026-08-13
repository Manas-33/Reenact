"""Evals: the scenario runner (a recording plus checks), plain-Python
assertions, and a trajectory-level LLM judge that scores multi-step behavior.
"""

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
from reenact.evals.suite import SuiteConfigError, load_suite

__all__ = [
    "Check",
    "CheckResult",
    "EvalReport",
    "JudgeVerdict",
    "RunView",
    "Scenario",
    "ScenarioResult",
    "SuiteConfigError",
    "answer_contains",
    "answer_matches",
    "called_tool",
    "did_not_call_tool",
    "judged",
    "load_suite",
    "no_mutating_tool_reexecuted",
    "render_trajectory",
    "replays_clean",
    "run_scenario",
    "run_suite",
    "tool_call_count",
]
