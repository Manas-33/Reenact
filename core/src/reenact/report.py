"""Render a regression diff into a PR comment, and post it without spamming.

The GitHub Action surfaces the gate two ways: a *sticky* PR comment updated in
place on every run (so a PR accrues one comment, never a pile), and a check-run
whose conclusion goes red on a regression. Rendering, the find-or-create sticky
logic, and the pass/fail mapping work against a duck-typed client; :class:`GitHubClient`
is the stdlib-only implementation, and its one network primitive is injectable, so
the request shapes are exercised with a fake and no token.
"""

import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, cast

from pydantic import BaseModel, ConfigDict

from reenact.evals.baseline import CheckDelta, RegressionDiff
from reenact.schema import LLMCallEvent, Trajectory

# A hidden marker - an HTML comment, invisible in rendered markdown - that lets a
# later run find its own previous comment and update it instead of posting a new one.
STICKY_MARKER = "<!-- reenact-gate -->"


def gate_marker(name: str = "") -> str:
    """The hidden sticky-comment marker for a gate, optionally scoped to an agent.

    An empty ``name`` keeps the shared default marker (one gate per repo); a name
    scopes it (``<!-- reenact-gate:support -->``) so several agents gated in one repo
    (a matrix) each keep their own comment instead of clobbering a single shared one.
    """
    return f"<!-- reenact-gate:{name} -->" if name else STICKY_MARKER


_FOOTER = (
    "<sub>Reenact replayed the recorded suite offline. $0, no network. It blocks the "
    "merge only on a regression versus the committed baseline.</sub>"
)

_TASK_MAX = 70
_ISSUE_PREFIX = re.compile(r"^\s*(issue\s*)?#\d+:\s*", re.IGNORECASE)
_CALL = re.compile(r"^(\w+)\((.*)\)$")


def _content_text(content: Any) -> str:
    """Flatten a message ``content`` (a string or a list of blocks) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in cast(list[Any], content):
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                mapping = cast(dict[str, Any], block)
                text = mapping.get("text")
                if mapping.get("type") == "text" and isinstance(text, str):
                    parts.append(text)
        return " ".join(parts)
    return ""


def scenario_task(trajectory: Trajectory) -> str:
    """A short human label for a run: the first line of its first user message.

    Strips a leading issue-number prefix and clips it, so a PR comment can show
    what each scenario is about without any extra config. Returns ``""`` if the run
    has no usable prompt, so the comment degrades to just the scenario name.
    """
    for event in trajectory.events:
        if not isinstance(event, LLMCallEvent):
            continue
        messages = event.request.get("messages")
        if not isinstance(messages, list):
            return ""
        for message in cast(list[Any], messages):
            if not isinstance(message, dict):
                continue
            mapping = cast(dict[str, Any], message)
            role = mapping.get("role") or mapping.get("type")
            if role not in ("user", "human"):
                continue
            text = _content_text(mapping.get("content")).strip()
            if not text:
                return ""
            line = _ISSUE_PREFIX.sub("", text.splitlines()[0]).strip()
            return line[:_TASK_MAX] + ("..." if len(line) > _TASK_MAX else "")
        return ""
    return ""


def _arg0(args: str) -> str:
    """The first argument of a rendered call label, unquoted (``'label_issue'`` ->
    ``label_issue``)."""
    return args.split(",")[0].strip().strip("'\"")


def _behavior(name: str) -> tuple[str, str, str]:
    """A check's (behavior label, passing value, failing value) for the diff table.

    The behavior names the dimension being compared, in plain English; the two
    values are what the Baseline and This-PR columns show for a boolean check (a
    scored check overrides them with its numbers). Falls back to a generic
    pass/fail for a custom check, so it is always safe.
    """
    call = _CALL.match(name)
    if call:
        func, arg = call.group(1), _arg0(call.group(2))
        if func == "called_tool":
            return f"calls `{arg}`", "yes", "no"
        if func == "did_not_call_tool":
            return f"calls `{arg}`", "no", "yes"
        if func == "answer_contains":
            return f"answer mentions '{arg}'", "yes", "no"
        if func == "answer_matches":
            return "answer matches the expected pattern", "yes", "no"
        if func == "tool_call_count":
            return f"calls `{arg}` the expected number of times", "yes", "no"
    if name.startswith("criterion:"):
        return f"criterion '{name.split(':', 1)[1]}'", "yes", "no"
    if name == "replays_clean":
        return "recording replays cleanly", "yes", "no"
    if name == "no_mutating_tool_reexecuted":
        return "mutating tools stay substituted", "yes", "no"
    return f"`{name}`", "pass", "fail"


def _scores(detail: str) -> tuple[str, str] | None:
    """The (baseline, current) numbers for a score-drop detail like ``0.91->0.62``."""
    parts = detail.split("->")
    if len(parts) != 2:
        return None
    try:
        float(parts[0])
        float(parts[1])
    except ValueError:
        return None
    return parts[0], parts[1]


def _baseline_current(delta: CheckDelta) -> tuple[str, str, str]:
    """A delta as (behavior, baseline value, this-PR value) for the diff table.

    A scored check shows its actual numbers; a boolean check shows the passing value
    (baseline) and the failing value (this PR), derived from what the check means.
    """
    behavior, pass_value, fail_value = _behavior(delta.check)
    scores = _scores(delta.detail)
    if scores is not None:
        return behavior, scores[0], scores[1]
    return behavior, pass_value, fail_value


def _cell(text: str) -> str:
    """Make text safe for a Markdown table cell (escape the column separator)."""
    return text.replace("|", "\\|")


def _scenario_sections(deltas: list[CheckDelta], tasks: dict[str, str]) -> list[str]:
    """Regressed checks grouped by scenario: a heading plus a small diff table.

    Each scenario becomes a heading (with its recorded task, when known); its
    flipped checks render as a Behavior / Baseline / This PR table, or a single line
    when only one check flipped. Grouping keeps the scenario named once instead of
    repeating it per check.
    """
    grouped: dict[str, list[CheckDelta]] = {}
    for delta in deltas:
        grouped.setdefault(delta.scenario, []).append(delta)

    lines: list[str] = []
    for scenario, group in grouped.items():
        header = f"**{scenario}**"
        task = tasks.get(scenario, "")
        if task:
            header += f" - *{task}*"
        lines += ["", header, ""]
        if len(group) == 1:
            behavior, baseline, current = _baseline_current(group[0])
            lines.append(f"- {behavior}: {baseline} -> **{current}**")
        else:
            lines += ["| Behavior | Baseline | This PR |", "| --- | --- | --- |"]
            for delta in group:
                behavior, baseline, current = _baseline_current(delta)
                lines.append(f"| {_cell(behavior)} | {baseline} | **{current}** |")
    return lines


def render_pr_comment(diff: RegressionDiff, *, marker: str = STICKY_MARKER) -> str:
    """Render the diff as the sticky PR comment body (leads with the marker).

    A regression leads with a plain-English headline, then one section per regressed
    scenario - the scenario (with its recorded task) as a heading, and its flipped
    checks as a Behavior / Baseline / This PR diff (a single line when only one
    flipped). A clean run says so plainly. Advisory warnings, improvements, and new
    checks follow when present. ``marker`` scopes the sticky comment to a named gate.
    """
    tasks = diff.scenario_tasks
    lines = [marker]
    if diff.regressed:
        n = len(diff.regressed_scenarios)
        lines += [
            "## Reenact: regression detected",
            "",
            f"This PR changed the agent's behavior. {n} of {diff.scenario_total} "
            "scenario(s) regressed, so the merge is blocked.",
            *_scenario_sections(diff.blocking_regressions, tasks),
            "",
            "If this change is intended, re-record the baseline. Otherwise it is a "
            "regression to fix before merging.",
        ]
    else:
        lines += [
            "## Reenact: no regressions",
            "",
            f"All {diff.scenario_total} scenario(s) match the committed baseline. "
            "Safe to merge.",
        ]
    if diff.advisory_regressions:
        lines += [
            "",
            "### Advisory (warnings only, not blocking)",
            *_scenario_sections(diff.advisory_regressions, tasks),
        ]
    if diff.improvements:
        improved = ", ".join(f"`{d.check}` in {d.scenario}" for d in diff.improvements)
        lines += ["", f"Improved: {improved}."]
    if diff.new_checks:
        added = ", ".join(f"`{d.check}` in {d.scenario}" for d in diff.new_checks)
        lines += ["", f"New checks (not gating): {added}."]
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


def post_report(
    client: GateClient, diff: RegressionDiff, *, head_sha: str, name: str = ""
) -> str:
    """Post the whole gate result: the sticky comment plus the check-run.

    Returns ``"created"`` / ``"updated"`` for the comment. The check-run's
    conclusion (red on a blocking regression) is what a required status check reads.
    ``name`` scopes the gate when several agents are gated in one repo: it names the
    check-run ``Reenact (<name>)`` and gives the comment its own marker, so each
    agent keeps a separate comment and check instead of clobbering one shared pair.
    An empty ``name`` keeps the single-gate default (``Reenact`` + the shared marker).
    """
    marker = gate_marker(name)
    check_name = f"{CHECK_RUN_NAME} ({name})" if name else CHECK_RUN_NAME
    action = upsert_sticky_comment(
        client, render_pr_comment(diff, marker=marker), marker=marker
    )
    result = check_run_result(diff)
    client.create_check_run(
        name=check_name,
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
