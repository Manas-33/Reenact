"""Calibration: per-criterion agreement between the structured evaluator and humans.

The structured evaluator (r3.6.5) answers criteria as booleans; this measures how
often those booleans match human judgment, criterion by criterion, and turns that
into the blocking/advisory level each criterion earns. A criterion the evaluator
agrees with humans on (at or above a threshold) is trusted to *block* the gate; a
noisier one is demoted to *advisory* - reported but never flaking the merge. So the
calibration number does not just get reported, it sets the gate's behavior - the
knob r3.6.5b left for exactly this.

Agreement is measured against the human signal two ways, always reported together:
evaluator-vs-human (is the evaluator as good as a person?) beside human-vs-human
(how much do people even agree?). An evaluator agreeing 85% is strong if people
agree 88% and suspicious if people agree 99% - so a per-criterion score above the
human-human ceiling, or above an absolute suspicious cutoff, is flagged rather than
celebrated (don't chase a suspicious number).

The evaluator is treated as just another rater, so everything is a homogeneous list
of :class:`Label`s and one agreement routine serves both comparisons. This module is
the mechanism, tested on synthetic labels; the real number needs a committed label
set from a second human rater over the r3.5 demo trajectories (the external
dependency) plus the evaluator run with a key - a later step.
"""

from itertools import combinations
from pathlib import Path
from statistics import mean

from pydantic import BaseModel, ConfigDict

from reenact.evals.check import CriterionLevel

DEFAULT_EVALUATOR = "evaluator"
DEFAULT_PROMOTE_THRESHOLD = 0.85
DEFAULT_AGREEMENT_FLOOR = 0.80
SUSPICIOUS_AGREEMENT = 0.92


class Label(BaseModel):
    """One judgment: a rater says a criterion passed or failed for a scenario.

    A human rater and the evaluator produce the same shape - the evaluator is just
    a rater whose id is ``DEFAULT_EVALUATOR`` - so one routine compares any two.
    """

    model_config = ConfigDict(extra="forbid")

    scenario: str
    criterion: str
    rater: str
    passed: bool


class LabelSet(BaseModel):
    """A committable collection of labels - human ratings plus evaluator runs."""

    model_config = ConfigDict(extra="forbid")

    labels: list[Label] = []


def save_label_set(label_set: LabelSet, path: Path) -> None:
    """Write a label set as deterministic JSON (stable order + trailing newline)."""
    path.write_text(label_set.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_label_set(path: Path) -> LabelSet:
    """Load a committed label set."""
    return LabelSet.model_validate_json(path.read_text(encoding="utf-8"))


class CriterionAgreement(BaseModel):
    """One criterion's agreement result and the level it earns."""

    model_config = ConfigDict(extra="forbid")

    criterion: str
    evaluator_agreement: float
    human_agreement: float | None  # None when fewer than two human raters
    n: int  # scenarios the evaluator agreement is computed over (the denominator)
    level: CriterionLevel
    within_floor: bool
    suspicious: bool


class CalibrationReport(BaseModel):
    """Per-criterion calibration, and the levels it assigns to the gate."""

    model_config = ConfigDict(extra="forbid")

    criteria: list[CriterionAgreement] = []

    def levels(self) -> dict[str, CriterionLevel]:
        """The blocking/advisory level each criterion earned, keyed by id."""
        return {c.criterion: c.level for c in self.criteria}


def _rate(matches: int, total: int) -> float:
    return round(matches / total, 4) if total else 0.0


def _by_criterion(labels: list[Label]) -> dict[str, dict[str, dict[str, bool]]]:
    """Index labels as ``criterion -> rater -> {scenario: passed}``."""
    index: dict[str, dict[str, dict[str, bool]]] = {}
    for label in labels:
        raters = index.setdefault(label.criterion, {})
        raters.setdefault(label.rater, {})[label.scenario] = label.passed
    return index


def _gold(human: dict[str, dict[str, bool]]) -> dict[str, bool]:
    """The adjudicated human label per scenario: majority vote, ties dropped."""
    scenarios = {s for judgments in human.values() for s in judgments}
    gold: dict[str, bool] = {}
    for scenario in scenarios:
        votes = [j[scenario] for j in human.values() if scenario in j]
        yes = sum(votes)
        no = len(votes) - yes
        if yes > no:
            gold[scenario] = True
        elif no > yes:
            gold[scenario] = False
        # a tie is genuinely ambiguous - it is not part of the gold set
    return gold


def _human_agreement(human: dict[str, dict[str, bool]]) -> float | None:
    """Mean pairwise agreement among human raters, or ``None`` if fewer than two."""
    rates: list[float] = []
    for rater_a, rater_b in combinations(sorted(human), 2):
        judgments_a, judgments_b = human[rater_a], human[rater_b]
        shared = set(judgments_a) & set(judgments_b)
        if not shared:
            continue
        matches = sum(1 for s in shared if judgments_a[s] == judgments_b[s])
        rates.append(matches / len(shared))
    return round(mean(rates), 4) if rates else None


def _criterion_agreement(
    criterion: str,
    raters: dict[str, dict[str, bool]],
    *,
    evaluator: str,
    promote_threshold: float,
    floor: float,
    suspicious_cutoff: float,
) -> CriterionAgreement:
    human = {r: j for r, j in raters.items() if r != evaluator}
    evaluator_judgments = raters.get(evaluator, {})
    gold = _gold(human)

    matches = sum(
        1
        for scenario, verdict in gold.items()
        if scenario in evaluator_judgments and evaluator_judgments[scenario] == verdict
    )
    total = sum(1 for scenario in gold if scenario in evaluator_judgments)
    evaluator_agreement = _rate(matches, total)
    human_agreement = _human_agreement(human)

    level = (
        CriterionLevel.BLOCKING
        if total and evaluator_agreement >= promote_threshold
        else CriterionLevel.ADVISORY
    )
    suspicious = total > 0 and (
        evaluator_agreement > suspicious_cutoff
        or (human_agreement is not None and evaluator_agreement > human_agreement)
    )
    return CriterionAgreement(
        criterion=criterion,
        evaluator_agreement=evaluator_agreement,
        human_agreement=human_agreement,
        n=total,
        level=level,
        within_floor=total > 0 and evaluator_agreement >= floor,
        suspicious=suspicious,
    )


def calibrate(
    labels: LabelSet | list[Label],
    *,
    evaluator: str = DEFAULT_EVALUATOR,
    promote_threshold: float = DEFAULT_PROMOTE_THRESHOLD,
    floor: float = DEFAULT_AGREEMENT_FLOOR,
    suspicious_cutoff: float = SUSPICIOUS_AGREEMENT,
) -> CalibrationReport:
    """Per-criterion evaluator-vs-human agreement and the level each criterion earns.

    ``labels`` mixes human ratings and the evaluator's own judgments (rater id
    ``evaluator``). For each criterion the evaluator's booleans are scored against
    the human majority gold (ties dropped), reported beside the human-human
    agreement, and a criterion at or above ``promote_threshold`` earns
    ``BLOCKING`` while the rest are ``ADVISORY``.
    """
    label_list = labels.labels if isinstance(labels, LabelSet) else labels
    index = _by_criterion(label_list)
    return CalibrationReport(
        criteria=[
            _criterion_agreement(
                criterion,
                index[criterion],
                evaluator=evaluator,
                promote_threshold=promote_threshold,
                floor=floor,
                suspicious_cutoff=suspicious_cutoff,
            )
            for criterion in sorted(index)
        ]
    )
