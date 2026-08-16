"""The eval runner: replay a scenario offline and run its checks.

A scenario's checks read a :class:`RunView` over the recording; the runner just
builds the view, evaluates each check, and aggregates the results into a
:class:`ScenarioResult` and, over a suite, an :class:`EvalReport`. Both are
serializable, so the CI gate can persist a baseline and diff a
later run against it.
"""

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

from reenact.evals.check import CheckResult, RunView
from reenact.evals.scenario import Scenario


class ScenarioResult(BaseModel):
    """The results of every check run against one scenario."""

    model_config = ConfigDict(extra="forbid")

    name: str
    checks: list[CheckResult] = []

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> list[CheckResult]:
        return [check for check in self.checks if not check.passed]


class EvalReport(BaseModel):
    """The results of a whole suite of scenarios."""

    model_config = ConfigDict(extra="forbid")

    scenarios: list[ScenarioResult] = []

    @property
    def passed(self) -> bool:
        return all(scenario.passed for scenario in self.scenarios)

    @property
    def total(self) -> int:
        return len(self.scenarios)

    @property
    def passed_count(self) -> int:
        return sum(1 for scenario in self.scenarios if scenario.passed)


def run_scenario(scenario: Scenario) -> ScenarioResult:
    """Evaluate every check in ``scenario`` against its recording."""
    view = RunView(scenario.trajectory)
    checks = [check(view) for check in scenario.checks]
    return ScenarioResult(name=scenario.name, checks=checks)


def run_suite(scenarios: Iterable[Scenario]) -> EvalReport:
    """Run every scenario and collect the results into one report."""
    return EvalReport(scenarios=[run_scenario(scenario) for scenario in scenarios])
