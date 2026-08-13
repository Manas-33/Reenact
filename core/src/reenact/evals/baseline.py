"""The regression baseline and the diff the CI gate reports.

A baseline is a committed snapshot of a suite's last-known-good results - each
scenario's checks reduced to ``(name, passed, score)``, no messages, so it diffs
cleanly in git and only changes when a check's outcome or score changes.
:func:`diff_baselines` compares a fresh run against it and reports what *regressed*
(a check that went pass to fail, or a judge score that dropped past a tolerance),
what improved, and what is new - the input the ``ci`` verb turns into an exit code.
"""

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from reenact.evals.runner import EvalReport

DEFAULT_SCORE_TOLERANCE = 0.05


class BaselineCheck(BaseModel):
    """One check's recorded outcome in a baseline."""

    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    score: float | None = None


class BaselineScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    checks: list[BaselineCheck] = []


class Baseline(BaseModel):
    """A committed snapshot of a suite's results, for the CI gate to diff against."""

    model_config = ConfigDict(extra="forbid")

    scenarios: list[BaselineScenario] = []

    @classmethod
    def from_report(cls, report: EvalReport) -> "Baseline":
        """Reduce a full eval report to the diff-friendly baseline (drop messages)."""
        return cls(
            scenarios=[
                BaselineScenario(
                    name=scenario.name,
                    checks=[
                        BaselineCheck(
                            name=check.name, passed=check.passed, score=check.score
                        )
                        for check in scenario.checks
                    ],
                )
                for scenario in report.scenarios
            ]
        )


def save_baseline(baseline: Baseline, path: Path) -> None:
    """Write ``baseline`` as deterministic JSON (stable order + trailing newline)."""
    path.write_text(baseline.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_baseline(path: Path) -> Baseline:
    """Load a committed baseline snapshot."""
    return Baseline.model_validate_json(path.read_text(encoding="utf-8"))


class DeltaKind(StrEnum):
    """How a check's outcome moved relative to the baseline."""

    REGRESSION = "regression"
    IMPROVEMENT = "improvement"
    NEW = "new"  # a check with no baseline counterpart - reported, does not gate


class CheckDelta(BaseModel):
    """One check whose outcome differs from the baseline."""

    model_config = ConfigDict(extra="forbid")

    scenario: str
    check: str
    kind: DeltaKind
    detail: str


def _change(prior: BaselineCheck, current: BaselineCheck) -> str:
    """A short 'before->after' string for a changed check."""
    if prior.score is not None and current.score is not None:
        return f"{prior.score:.2f}->{current.score:.2f}"
    before = "pass" if prior.passed else "fail"
    after = "pass" if current.passed else "fail"
    return f"{before}->{after}"


def _classify(
    prior: BaselineCheck, current: BaselineCheck, tolerance: float
) -> DeltaKind | None:
    """Regression, improvement, or unchanged (``None``) for one check pair.

    A pass to fail is a regression whatever the score; otherwise a judge score
    that moved past ``tolerance`` in either direction is the signal. Everything
    else is unchanged.
    """
    if prior.passed and not current.passed:
        return DeltaKind.REGRESSION
    if not prior.passed and current.passed:
        return DeltaKind.IMPROVEMENT
    if prior.score is not None and current.score is not None:
        if current.score < prior.score - tolerance:
            return DeltaKind.REGRESSION
        if current.score > prior.score + tolerance:
            return DeltaKind.IMPROVEMENT
    return None


class RegressionDiff(BaseModel):
    """The outcome of diffing a fresh run against a baseline."""

    model_config = ConfigDict(extra="forbid")

    deltas: list[CheckDelta] = []
    scenario_total: int = 0

    @property
    def regressions(self) -> list[CheckDelta]:
        return [d for d in self.deltas if d.kind is DeltaKind.REGRESSION]

    @property
    def improvements(self) -> list[CheckDelta]:
        return [d for d in self.deltas if d.kind is DeltaKind.IMPROVEMENT]

    @property
    def new_checks(self) -> list[CheckDelta]:
        return [d for d in self.deltas if d.kind is DeltaKind.NEW]

    @property
    def regressed_scenarios(self) -> list[str]:
        ordered: list[str] = []
        for delta in self.regressions:
            if delta.scenario not in ordered:
                ordered.append(delta.scenario)
        return ordered

    @property
    def regressed(self) -> bool:
        return bool(self.regressions)

    def summary(self) -> str:
        """A one-line headline in the launch voice."""
        if not self.regressed:
            return f"no regressions across {self.scenario_total} scenario(s)"
        details = "; ".join(
            f"{d.check} {d.detail} in {d.scenario}" for d in self.regressions
        )
        return (
            f"{len(self.regressed_scenarios)}/{self.scenario_total} "
            f"scenarios regressed: {details}"
        )


def diff_baselines(
    baseline: Baseline,
    current: Baseline,
    *,
    score_tolerance: float = DEFAULT_SCORE_TOLERANCE,
) -> RegressionDiff:
    """Diff a fresh run (``current``) against ``baseline``.

    Matches checks by (scenario name, check name). A check present now but not in
    the baseline is ``NEW`` (reported, never a gate failure); a check that
    regressed or improved is recorded; unchanged checks are omitted.
    """
    prior_index = {
        (scenario.name, check.name): check
        for scenario in baseline.scenarios
        for check in scenario.checks
    }
    deltas: list[CheckDelta] = []
    for scenario in current.scenarios:
        for check in scenario.checks:
            prior = prior_index.get((scenario.name, check.name))
            if prior is None:
                deltas.append(
                    CheckDelta(
                        scenario=scenario.name,
                        check=check.name,
                        kind=DeltaKind.NEW,
                        detail="new",
                    )
                )
                continue
            kind = _classify(prior, check, score_tolerance)
            if kind is not None:
                deltas.append(
                    CheckDelta(
                        scenario=scenario.name,
                        check=check.name,
                        kind=kind,
                        detail=_change(prior, check),
                    )
                )
    return RegressionDiff(deltas=deltas, scenario_total=len(current.scenarios))
