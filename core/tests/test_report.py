"""Rendering the regression diff into a sticky PR comment + check-run, and posting it.

Keyless throughout: a fake comment client for the render/sticky logic, and a fake
HTTP transport for the real GitHubClient so its request shapes are checked without a
network or a token. The render carries the hidden marker, a clean vs regressed run
differ, the sticky logic updates its own prior comment in place (never spams) and
never touches a human's comment, the check-run goes red only on a regression, and
the client builds the right REST calls.
"""

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from reenact.cli import app
from reenact.evals import Baseline, CriterionLevel, diff_baselines
from reenact.evals.baseline import BaselineCheck, BaselineScenario, RegressionDiff
from reenact.report import (
    STICKY_MARKER,
    CheckConclusion,
    GitHubClient,
    IssueComment,
    check_run_result,
    post_report,
    render_pr_comment,
    scenario_task,
    upsert_sticky_comment,
)
from reenact.schema import LLMCallEvent, Trajectory

runner = CliRunner()


def _check(
    name: str, passed: bool, level: CriterionLevel = CriterionLevel.BLOCKING
) -> BaselineCheck:
    return BaselineCheck(name=name, passed=passed, level=level)


def _diff(
    before: BaselineCheck,
    after: BaselineCheck,
    *,
    tasks: dict[str, str] | None = None,
) -> RegressionDiff:
    base = Baseline(scenarios=[BaselineScenario(name="weather", checks=[before])])
    now = Baseline(scenarios=[BaselineScenario(name="weather", checks=[after])])
    return diff_baselines(base, now, scenario_tasks=tasks)


def _diff_checks(
    scenario: str,
    pairs: list[tuple[BaselineCheck, BaselineCheck]],
    *,
    tasks: dict[str, str] | None = None,
) -> RegressionDiff:
    """A diff for one scenario with several checks (before, after) each."""
    base = Baseline(
        scenarios=[BaselineScenario(name=scenario, checks=[b for b, _ in pairs])]
    )
    now = Baseline(
        scenarios=[BaselineScenario(name=scenario, checks=[a for _, a in pairs])]
    )
    return diff_baselines(base, now, scenario_tasks=tasks)


class _FakeClient:
    """A fake GateClient: records comment ops and check-runs, no network."""

    def __init__(self, comments: list[IssueComment] | None = None) -> None:
        self._comments = comments or []
        self.created: list[str] = []
        self.updated: list[tuple[int, str]] = []
        self.check_runs: list[dict[str, str]] = []

    def list_comments(self) -> list[IssueComment]:
        return list(self._comments)

    def create_comment(self, body: str) -> None:
        self.created.append(body)

    def update_comment(self, comment_id: int, body: str) -> None:
        self.updated.append((comment_id, body))

    def create_check_run(
        self, *, name: str, head_sha: str, conclusion: str, title: str, summary: str
    ) -> None:
        self.check_runs.append(
            {"name": name, "head_sha": head_sha, "conclusion": conclusion}
        )


class _FakeTransport:
    """Records each request and returns queued (status, payload) responses."""

    def __init__(self, queue: list[tuple[int, object]] | None = None) -> None:
        self.queue = queue or []
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> tuple[int, bytes]:
        self.calls.append(
            {"method": method, "url": url, "headers": headers, "body": body}
        )
        status, payload = self.queue.pop(0) if self.queue else (200, None)
        data = json.dumps(payload).encode("utf-8") if payload is not None else b""
        return status, data


# --- rendering ---------------------------------------------------------------


def test_regressed_comment_groups_checks_under_a_scenario() -> None:
    diff = _diff_checks(
        "refund-final-sale",
        [
            (
                _check("did_not_call_tool('issue_refund')", True),
                _check("did_not_call_tool('issue_refund')", False),
            ),
            (
                _check("answer_contains('not eligible')", True),
                _check("answer_contains('not eligible')", False),
            ),
        ],
        tasks={"refund-final-sale": "refund for order 1002, the festival ticket"},
    )
    body = render_pr_comment(diff)
    assert body.startswith(STICKY_MARKER)
    assert "regression detected" in body
    assert "**refund-final-sale**" in body  # scenario as a heading...
    assert body.count("**refund-final-sale**") == 1  # ...named once, not per check
    assert "refund for order 1002" in body  # the recorded task, surfaced
    # A Behavior / Baseline / This PR mini-table when 2+ checks flip.
    assert "| Behavior | Baseline | This PR |" in body
    assert "| calls `issue_refund` | no | **yes** |" in body
    assert "| answer mentions 'not eligible' | yes | **no** |" in body
    assert "pass->fail" not in body  # the terse detail is gone
    assert "—" not in body  # no em-dashes


def test_single_flip_renders_as_a_one_liner() -> None:
    diff = _diff(
        _check("called_tool('x')", True),
        _check("called_tool('x')", False),
        tasks={"weather": "What's the weather in Paris?"},
    )
    body = render_pr_comment(diff)
    assert "**weather**" in body
    assert "What's the weather in Paris?" in body
    assert "- calls `x`: yes -> **no**" in body  # one line, not a table
    assert "| Behavior |" not in body  # no mini-table for a single flip
    assert "pass->fail" not in body


def test_scored_check_shows_the_numbers() -> None:
    before = BaselineCheck(name="criterion:grounded", passed=True, score=0.9)
    after = BaselineCheck(name="criterion:grounded", passed=True, score=0.6)
    body = render_pr_comment(_diff(before, after))  # score drop past tolerance
    assert "criterion 'grounded': 0.90 -> **0.60**" in body


def test_clean_and_regressed_comments_differ() -> None:
    clean = render_pr_comment(_diff(_check("c", True), _check("c", True)))
    regressed = render_pr_comment(_diff(_check("c", True), _check("c", False)))
    assert "no regressions" in clean and "Safe to merge" in clean
    assert "regression detected" in regressed
    assert STICKY_MARKER in clean and STICKY_MARKER in regressed


def test_comment_degrades_without_a_task() -> None:
    diff = _diff(_check("called_tool('x')", True), _check("called_tool('x')", False))
    body = render_pr_comment(diff)
    assert "**weather**" in body  # scenario heading still shown
    assert "**weather** -" not in body  # no task appended when none is available


def test_advisory_regression_shown_as_non_blocking() -> None:
    adv = CriterionLevel.ADVISORY
    diff = _diff(
        _check("criterion:tone", True, adv), _check("criterion:tone", False, adv)
    )
    body = render_pr_comment(diff)
    # Advisory flip is a warning section, and the comment does not claim a block.
    assert "Advisory (warnings only, not blocking)" in body
    assert "no regressions" in body  # nothing blocking


def test_scenario_task_reads_clips_and_strips_prefix() -> None:
    trajectory = Trajectory(
        name="t",
        events=[
            LLMCallEvent(
                seq=0,
                provider="anthropic",
                model="m",
                request={
                    "messages": [
                        {
                            "role": "user",
                            "content": "Issue #42: Password reset link error\n\nbody",
                        }
                    ]
                },
                response={"content": [{"type": "text", "text": "ok"}]},
                request_hash="h",
            )
        ],
    )
    assert scenario_task(trajectory) == "Password reset link error"


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


# --- post_report (whole gate result over a fake client) ----------------------


def test_post_report_comments_and_opens_a_red_check_run() -> None:
    client = _FakeClient()
    diff = _diff(_check("called_tool('x')", True), _check("called_tool('x')", False))
    action = post_report(client, diff, head_sha="abc123")
    assert action == "created"
    assert len(client.created) == 1
    assert client.check_runs == [
        {"name": "Reenact", "head_sha": "abc123", "conclusion": "failure"}
    ]


def test_post_report_green_check_run_when_clean() -> None:
    client = _FakeClient([IssueComment(id=9, body=f"{STICKY_MARKER} old")])
    diff = _diff(_check("c", True), _check("c", True))
    action = post_report(client, diff, head_sha="def456")
    assert action == "updated"  # edits its own prior comment
    assert client.check_runs[0]["conclusion"] == "success"


# --- the real GitHubClient's request shapes (fake transport) -----------------


def test_client_list_comments_gets_and_parses() -> None:
    transport = _FakeTransport(
        [(200, [{"id": 5, "body": "hi"}, {"id": 6, "body": "there"}])]
    )
    client = GitHubClient(repo="o/r", issue_number=3, token="tok", transport=transport)
    assert client.list_comments() == [
        IssueComment(id=5, body="hi"),
        IssueComment(id=6, body="there"),
    ]
    call = transport.calls[0]
    assert call["method"] == "GET"
    assert call["url"] == "https://api.github.com/repos/o/r/issues/3/comments"
    assert call["headers"]["Authorization"] == "Bearer tok"


def test_client_check_run_posts_the_right_payload() -> None:
    transport = _FakeTransport([(201, {"id": 1})])
    client = GitHubClient(repo="o/r", issue_number=3, token="t", transport=transport)
    client.create_check_run(
        name="Reenact", head_sha="abc", conclusion="failure", title="T", summary="S"
    )
    call = transport.calls[0]
    assert call["method"] == "POST"
    assert call["url"].endswith("/repos/o/r/check-runs")
    payload = json.loads(call["body"])
    assert payload["head_sha"] == "abc"
    assert payload["conclusion"] == "failure"
    assert payload["status"] == "completed"
    assert payload["output"] == {"title": "T", "summary": "S"}


def test_client_raises_on_api_error() -> None:
    transport = _FakeTransport([(404, {"message": "Not Found"})])
    client = GitHubClient(repo="o/r", issue_number=3, token="t", transport=transport)
    try:
        client.list_comments()
    except RuntimeError as exc:
        assert "404" in str(exc)
    else:  # pragma: no cover - the call must raise
        raise AssertionError("expected a RuntimeError on a 404")


# --- the `reenact report` command --------------------------------------------


def _diff_file(tmp_path: Path, *, regressed: bool) -> Path:
    after = _check("c", not regressed)
    diff = _diff(_check("c", True), after)
    path = tmp_path / "diff.json"
    path.write_text(diff.model_dump_json(), encoding="utf-8")
    return path


def test_report_skips_on_missing_diff(tmp_path: Path) -> None:
    # A config error exits `ci` before writing the diff; the post step must not crash.
    args = ["report", str(tmp_path / "absent.json")]
    args += ["--repo", "o/r", "--pr", "1", "--token", "t"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0
    assert "no diff file" in result.stdout


def test_report_skips_without_a_token(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    path = _diff_file(tmp_path, regressed=True)
    result = runner.invoke(app, ["report", str(path)])
    assert result.exit_code == 0
    assert "skipping" in result.stdout


def test_report_posts_on_the_happy_path(tmp_path: Path, monkeypatch: Any) -> None:
    posted: dict[str, Any] = {}

    class _StubClient:
        def __init__(self, **kwargs: Any) -> None:
            posted["init"] = kwargs

        def list_comments(self) -> list[IssueComment]:
            return []

        def create_comment(self, body: str) -> None:
            posted["comment"] = body

        def update_comment(self, comment_id: int, body: str) -> None:
            posted["updated"] = (comment_id, body)

        def create_check_run(self, **kwargs: str) -> None:
            posted["check_run"] = kwargs

    monkeypatch.setattr("reenact.cli.GitHubClient", _StubClient)
    path = _diff_file(tmp_path, regressed=True)
    args = ["report", str(path)]
    args += ["--repo", "o/r", "--pr", "7", "--sha", "s", "--token", "t"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stdout
    assert posted["init"] == {"repo": "o/r", "issue_number": 7, "token": "t"}
    assert STICKY_MARKER in posted["comment"]
    assert posted["check_run"]["conclusion"] == "failure"
    assert "check-run red" in result.stdout
