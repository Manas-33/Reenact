"""Rendering the regression diff into a sticky PR comment + check-run.

Keyless, with a fake comment client: the render carries the hidden marker and the
headline, a clean run and a regressed run differ, the sticky logic updates its own
prior comment in place (never spams) and never touches a human's comment, and the
check-run conclusion goes red only on a regression.
"""

from reenact.evals import Baseline, CriterionLevel, diff_baselines
from reenact.evals.baseline import BaselineCheck, BaselineScenario, RegressionDiff
from reenact.report import (
    STICKY_MARKER,
    CheckConclusion,
    IssueComment,
    check_run_result,
    render_pr_comment,
    upsert_sticky_comment,
)


def _check(
    name: str, passed: bool, level: CriterionLevel = CriterionLevel.BLOCKING
) -> BaselineCheck:
    return BaselineCheck(name=name, passed=passed, level=level)


def _diff(before: BaselineCheck, after: BaselineCheck) -> RegressionDiff:
    base = Baseline(scenarios=[BaselineScenario(name="weather", checks=[before])])
    now = Baseline(scenarios=[BaselineScenario(name="weather", checks=[after])])
    return diff_baselines(base, now)


class _FakeClient:
    def __init__(self, comments: list[IssueComment] | None = None) -> None:
        self._comments = comments or []
        self.created: list[str] = []
        self.updated: list[tuple[int, str]] = []

    def list_comments(self) -> list[IssueComment]:
        return list(self._comments)

    def create_comment(self, body: str) -> None:
        self.created.append(body)

    def update_comment(self, comment_id: int, body: str) -> None:
        self.updated.append((comment_id, body))


# --- rendering ---------------------------------------------------------------


def test_regressed_comment_carries_marker_and_details() -> None:
    diff = _diff(_check("called_tool('x')", True), _check("called_tool('x')", False))
    body = render_pr_comment(diff)
    assert body.startswith(STICKY_MARKER)
    assert "regression detected" in body
    assert "these block the merge" in body
    assert "`called_tool('x')` pass->fail - weather" in body


def test_clean_and_regressed_comments_differ() -> None:
    clean = render_pr_comment(_diff(_check("c", True), _check("c", True)))
    regressed = render_pr_comment(_diff(_check("c", True), _check("c", False)))
    assert "no regressions" in clean
    assert "regression detected" in regressed
    assert STICKY_MARKER in clean and STICKY_MARKER in regressed


def test_advisory_regression_shown_as_non_blocking() -> None:
    adv = CriterionLevel.ADVISORY
    diff = _diff(
        _check("criterion:tone", True, adv), _check("criterion:tone", False, adv)
    )
    body = render_pr_comment(diff)
    # Advisory flip is reported as a warning, and the comment does not claim a block.
    assert "warnings, do not block" in body
    assert "no regressions" in body  # nothing blocking


# --- sticky upsert -----------------------------------------------------------


def test_upsert_creates_when_no_prior_comment() -> None:
    client = _FakeClient()
    action = upsert_sticky_comment(client, "hello")
    assert action == "created"
    assert client.created == ["hello"]
    assert client.updated == []


def test_upsert_updates_its_own_prior_comment() -> None:
    prior = IssueComment(id=7, body=f"{STICKY_MARKER}\n## old")
    client = _FakeClient([prior])
    action = upsert_sticky_comment(client, "fresh")
    assert action == "updated"
    assert client.updated == [(7, "fresh")]
    assert client.created == []  # no new comment - the never-spams guarantee


def test_upsert_ignores_a_human_comment() -> None:
    human = IssueComment(id=1, body="looks good to me!")
    client = _FakeClient([human])
    upsert_sticky_comment(client, "gate result")
    # The human comment is untouched; a fresh gate comment is created.
    assert client.updated == []
    assert client.created == ["gate result"]


# --- check-run ---------------------------------------------------------------


def test_check_run_red_on_regression_green_otherwise() -> None:
    regressed = check_run_result(_diff(_check("c", True), _check("c", False)))
    assert regressed.conclusion is CheckConclusion.FAILURE
    assert "Regression" in regressed.title

    clean = check_run_result(_diff(_check("c", True), _check("c", True)))
    assert clean.conclusion is CheckConclusion.SUCCESS
    assert "No regressions" in clean.title
