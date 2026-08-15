"""Evaluator calibration - per-criterion agreement, and the levels it assigns.

Synthetic labels (keyless): the evaluator is just another rater, so each test hand-
builds a small set of human ratings plus evaluator judgments and asserts the
agreement math and the promotion. A criterion the evaluator matches humans on earns
`blocking`; a shaky one is demoted to `advisory`; agreement is reported beside the
human-human baseline, ties are dropped from the gold, and a suspiciously-high score
is flagged. The real >=100-label number over the demo corpus is a later step.
"""

from pathlib import Path

from reenact.evals import (
    CalibrationReport,
    CriterionAgreement,
    CriterionLevel,
    Label,
    LabelSet,
    calibrate,
    load_label_set,
    save_label_set,
)


def _labels(
    criterion: str,
    *,
    evaluator: dict[str, bool],
    raters: dict[str, dict[str, bool]],
) -> list[Label]:
    """Build labels for one criterion: an evaluator row plus named human raters."""
    out = [
        Label(scenario=s, criterion=criterion, rater="evaluator", passed=p)
        for s, p in evaluator.items()
    ]
    for rater, judgments in raters.items():
        out += [
            Label(scenario=s, criterion=criterion, rater=rater, passed=p)
            for s, p in judgments.items()
        ]
    return out


def _only(report: CalibrationReport, criterion: str) -> CriterionAgreement:
    """Pull the single CriterionAgreement out of a report by its id."""
    matches = [c for c in report.criteria if c.criterion == criterion]
    assert len(matches) == 1
    return matches[0]


# --- promotion ---------------------------------------------------------------


def test_high_agreement_promotes_to_blocking() -> None:
    # Evaluator matches the (agreeing) humans on all four scenarios -> 1.0.
    both = {"s1": True, "s2": True, "s3": False, "s4": True}
    report = calibrate(
        _labels("grounded", evaluator=both, raters={"alice": both, "bob": both})
    )
    result = _only(report, "grounded")
    assert result.evaluator_agreement == 1.0
    assert result.human_agreement == 1.0
    assert result.n == 4
    assert result.level is CriterionLevel.BLOCKING
    assert report.levels() == {"grounded": CriterionLevel.BLOCKING}


def test_shaky_agreement_demotes_to_advisory() -> None:
    gold = {"s1": True, "s2": True, "s3": True, "s4": True, "s5": True}
    # The evaluator disagrees on three of five -> 0.4, below the 0.85 threshold.
    ev = {"s1": True, "s2": True, "s3": False, "s4": False, "s5": False}
    report = calibrate(
        _labels("style", evaluator=ev, raters={"alice": gold, "bob": gold})
    )
    result = _only(report, "style")
    assert result.evaluator_agreement == 0.4
    assert result.level is CriterionLevel.ADVISORY
    assert not result.within_floor


# --- human-human baseline and gold -------------------------------------------


def test_human_disagreement_drops_the_tie_from_gold() -> None:
    # Alice and Bob disagree on s5 (a tie with two raters) -> s5 is not in the gold,
    # so the evaluator is scored over the four they agree on, and its s5 call does
    # not count. human_agreement is still over all five they both rated.
    alice = {"s1": True, "s2": True, "s3": False, "s4": True, "s5": True}
    bob = {"s1": True, "s2": True, "s3": False, "s4": True, "s5": False}
    ev = {"s1": True, "s2": True, "s3": False, "s4": True, "s5": True}
    report = calibrate(_labels("c", evaluator=ev, raters={"alice": alice, "bob": bob}))
    result = _only(report, "c")
    assert result.n == 4  # s5 dropped from gold
    assert result.evaluator_agreement == 1.0
    assert result.human_agreement == 0.8  # 4/5


def test_single_human_has_no_human_agreement() -> None:
    gold = {"s1": True, "s2": False, "s3": True}
    report = calibrate(_labels("c", evaluator=gold, raters={"alice": gold}))
    result = _only(report, "c")
    assert result.human_agreement is None
    assert result.evaluator_agreement == 1.0


# --- suspicious flag ---------------------------------------------------------


def test_agreement_above_human_ceiling_is_suspicious() -> None:
    # Humans agree only 0.6, but the evaluator "agrees" 1.0 with the gold - beating
    # the human ceiling is suspicious even though 1.0 also trips the absolute cutoff.
    alice = {"s1": True, "s2": True, "s3": True, "s4": True, "s5": True}
    bob = {"s1": True, "s2": True, "s3": True, "s4": False, "s5": False}
    # Gold keeps only the three they agree on; evaluator matches all three.
    ev = {"s1": True, "s2": True, "s3": True, "s4": True, "s5": True}
    report = calibrate(_labels("c", evaluator=ev, raters={"alice": alice, "bob": bob}))
    result = _only(report, "c")
    assert result.human_agreement == 0.6  # 3/5
    assert result.suspicious


def test_moderate_agreement_is_not_suspicious() -> None:
    gold = {"s1": True, "s2": True, "s3": True, "s4": True, "s5": False}
    # Evaluator agrees 4/5 = 0.8; humans agree perfectly (1.0), so 0.8 is below the
    # ceiling and below the absolute cutoff - trusted-ish but not flagged.
    ev = {"s1": True, "s2": True, "s3": True, "s4": True, "s5": True}
    report = calibrate(_labels("c", evaluator=ev, raters={"alice": gold, "bob": gold}))
    result = _only(report, "c")
    assert result.evaluator_agreement == 0.8
    assert not result.suspicious
    assert result.within_floor  # 0.8 >= 0.80 floor


# --- edges and persistence ---------------------------------------------------


def test_no_evaluator_labels_is_safe() -> None:
    # Humans labelled, the evaluator did not run this criterion -> n 0, advisory.
    gold = {"s1": True, "s2": False}
    report = calibrate(_labels("c", evaluator={}, raters={"alice": gold, "bob": gold}))
    result = _only(report, "c")
    assert result.n == 0
    assert result.evaluator_agreement == 0.0
    assert result.level is CriterionLevel.ADVISORY
    assert not result.suspicious


def test_multiple_criteria_each_get_a_level() -> None:
    good = {"s1": True, "s2": True, "s3": True}
    report = calibrate(
        _labels("trusted", evaluator=good, raters={"a": good, "b": good})
        + _labels(
            "shaky",
            evaluator={"s1": False, "s2": False, "s3": False},
            raters={"a": good, "b": good},
        )
    )
    assert report.levels() == {
        "trusted": CriterionLevel.BLOCKING,
        "shaky": CriterionLevel.ADVISORY,
    }


def test_label_set_round_trips(tmp_path: Path) -> None:
    label_set = LabelSet(
        labels=[
            Label(scenario="s1", criterion="c", rater="alice", passed=True),
            Label(scenario="s1", criterion="c", rater="evaluator", passed=False),
        ]
    )
    path = tmp_path / "labels.json"
    save_label_set(label_set, path)
    assert load_label_set(path) == label_set
    # deterministic: same content re-serializes byte-for-byte
    first = path.read_text(encoding="utf-8")
    save_label_set(load_label_set(path), path)
    assert path.read_text(encoding="utf-8") == first
