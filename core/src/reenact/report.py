"""Render a regression diff into a PR comment, and post it without spamming.

The GitHub Action surfaces the gate two ways: a *sticky* PR comment updated in
place on every run (so a PR accrues one comment, never a pile), and a check-run
whose conclusion goes red on a regression. This module is the testable core of
both - the markdown rendering, the find-or-create sticky logic (against a
duck-typed client), and the pass/fail mapping. The real GitHub HTTP client and the
action wiring are a later rung; here a fake client proves the mechanism offline,
the same rule as the recorder and judge adapters.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from reenact.evals.baseline import CheckDelta, RegressionDiff

# A hidden marker - an HTML comment, invisible in rendered markdown - that lets a
# later run find its own previous comment and update it instead of posting a new one.
STICKY_MARKER = "<!-- reenact-gate -->"

_FOOTER = (
    "<sub>Reenact replayed the recorded suite offline ($0, no network). It blocks "
    "the merge only on a regression versus the committed baseline.</sub>"
)


def _bullets(
    title: str, deltas: list[CheckDelta], *, with_detail: bool = True
) -> list[str]:
    if not deltas:
        return []
    lines = ["", f"**{title}**"]
    for delta in deltas:
        detail = f" {delta.detail}" if with_detail else ""
        lines.append(f"- `{delta.check}`{detail} - {delta.scenario}")
    return lines


def render_pr_comment(diff: RegressionDiff) -> str:
    """Render the diff as the sticky PR comment body (leads with the marker)."""
    heading = (
        "Reenact - regression detected"
        if diff.regressed
        else "Reenact - no regressions"
    )
    lines = [STICKY_MARKER, f"## {heading}", "", diff.summary()]
    lines += _bullets("Regressions (these block the merge):", diff.blocking_regressions)
    lines += _bullets("Advisory (warnings, do not block):", diff.advisory_regressions)
    lines += _bullets("Improvements:", diff.improvements)
    lines += _bullets(
        "New checks (reported, do not gate):", diff.new_checks, with_detail=False
    )
    lines += ["", _FOOTER]
    return "\n".join(lines)


@dataclass
class IssueComment:
    """The pieces of a PR comment the sticky logic needs."""

    id: int
    body: str


class CommentClient(Protocol):
    """A duck-typed GitHub issue-comment client (the real one is a later rung)."""

    def list_comments(self) -> list[IssueComment]: ...
    def create_comment(self, body: str) -> None: ...
    def update_comment(self, comment_id: int, body: str) -> None: ...


def upsert_sticky_comment(
    client: CommentClient, body: str, *, marker: str = STICKY_MARKER
) -> str:
    """Update this gate's existing comment in place, or create it if there is none.

    Finds the first comment carrying ``marker`` (a prior run's own comment) and
    edits it; otherwise posts a new one. So a PR gets exactly one gate comment no
    matter how many times CI runs - the "never spams" guarantee - and a human's
    comments are never touched. Returns ``"updated"`` or ``"created"``.
    """
    for comment in client.list_comments():
        if marker in comment.body:
            client.update_comment(comment.id, body)
            return "updated"
    client.create_comment(body)
    return "created"


class CheckConclusion(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"


class CheckRunResult(BaseModel):
    """The fields for the GitHub check-run that gates the merge."""

    model_config = ConfigDict(extra="forbid")

    conclusion: CheckConclusion
    title: str
    summary: str


def check_run_result(diff: RegressionDiff) -> CheckRunResult:
    """Map the diff to a check-run: red on a (blocking) regression, green otherwise."""
    if diff.regressed:
        return CheckRunResult(
            conclusion=CheckConclusion.FAILURE,
            title="Regression detected",
            summary=diff.summary(),
        )
    return CheckRunResult(
        conclusion=CheckConclusion.SUCCESS,
        title="No regressions",
        summary=diff.summary(),
    )
