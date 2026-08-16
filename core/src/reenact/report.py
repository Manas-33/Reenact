"""Render a regression diff into a PR comment, and post it without spamming.

The GitHub Action surfaces the gate two ways: a *sticky* PR comment updated in
place on every run (so a PR accrues one comment, never a pile), and a check-run
whose conclusion goes red on a regression. Rendering, the find-or-create sticky
logic, and the pass/fail mapping work against a duck-typed client; :class:`GitHubClient`
is the stdlib-only implementation, and its one network primitive is injectable, so
the request shapes are exercised with a fake and no token.
"""

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, cast

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
    """A duck-typed GitHub issue-comment client; :class:`GitHubClient` implements it."""

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


CHECK_RUN_NAME = "Reenact"


class GateClient(CommentClient, Protocol):
    """A comment client that can also open the merge-gating check-run."""

    def create_check_run(
        self, *, name: str, head_sha: str, conclusion: str, title: str, summary: str
    ) -> None: ...


def post_report(client: GateClient, diff: RegressionDiff, *, head_sha: str) -> str:
    """Post the whole gate result: the sticky comment plus the check-run.

    Returns ``"created"`` / ``"updated"`` for the comment. The check-run's
    conclusion (red on a blocking regression) is what a required status check reads.
    """
    action = upsert_sticky_comment(client, render_pr_comment(diff))
    result = check_run_result(diff)
    client.create_check_run(
        name=CHECK_RUN_NAME,
        head_sha=head_sha,
        conclusion=result.conclusion.value,
        title=result.title,
        summary=result.summary,
    )
    return action


# A transport is the one network primitive: (method, url, headers, body) -> (status,
# bytes). The real one uses urllib; a test injects a fake, so request-building is
# checked offline while the actual send stays a thin, untested seam.
Transport = Callable[[str, str, dict[str, str], bytes | None], tuple[int, bytes]]


def _urllib_transport(
    method: str, url: str, headers: dict[str, str], body: bytes | None
) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        # urlopen raises on 4xx/5xx; hand the status + body back to the caller.
        return error.code, error.read()


class GitHubClient:
    """A minimal GitHub REST client for the gate's comment + check-run.

    stdlib-only (no ``requests`` dependency), the same dependency-light rule as the
    rest of reenact. The HTTP send is injectable (``transport``) so the request
    shapes are unit-tested without a network or a token.
    """

    def __init__(
        self,
        *,
        repo: str,
        issue_number: int,
        token: str,
        api_base: str = "https://api.github.com",
        transport: Transport | None = None,
    ) -> None:
        self._repo = repo
        self._number = issue_number
        self._token = token
        self._api_base = api_base.rstrip("/")
        self._transport = transport or _urllib_transport

    def _request(self, method: str, path: str, payload: object = None) -> Any:
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "reenact",
        }
        body: bytes | None = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        status, data = self._transport(method, f"{self._api_base}{path}", headers, body)
        if status >= 400:
            detail = data.decode("utf-8", "replace")[:200]
            raise RuntimeError(f"GitHub API {method} {path} -> {status}: {detail}")
        return json.loads(data) if data else None

    def list_comments(self) -> list[IssueComment]:
        data = self._request(
            "GET", f"/repos/{self._repo}/issues/{self._number}/comments"
        )
        if not isinstance(data, list):
            return []
        return [
            IssueComment(id=int(item["id"]), body=str(item["body"]))
            for item in cast(list[dict[str, Any]], data)
        ]

    def create_comment(self, body: str) -> None:
        self._request(
            "POST",
            f"/repos/{self._repo}/issues/{self._number}/comments",
            {"body": body},
        )

    def update_comment(self, comment_id: int, body: str) -> None:
        self._request(
            "PATCH",
            f"/repos/{self._repo}/issues/comments/{comment_id}",
            {"body": body},
        )

    def create_check_run(
        self, *, name: str, head_sha: str, conclusion: str, title: str, summary: str
    ) -> None:
        self._request(
            "POST",
            f"/repos/{self._repo}/check-runs",
            {
                "name": name,
                "head_sha": head_sha,
                "status": "completed",
                "conclusion": conclusion,
                "output": {"title": title, "summary": summary},
            },
        )
