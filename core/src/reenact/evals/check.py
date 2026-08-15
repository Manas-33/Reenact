"""Checks: plain-Python assertions over a replayed run.

A check is a function ``(RunView) -> CheckResult``. The :class:`RunView` is a
read-only accessor over a recorded trajectory - the final answer, the tool calls,
the LLM calls - so an assertion reads like ordinary Python (``answer_contains``,
``called_tool``) rather than poking at raw event dicts. Each assertion here is a
factory that bakes its arguments into such a function; scenarios compose them.

Two checks reach into the replay engine: :func:`replays_clean` confirms the
recording reproduces offline with no divergence (a self-consistency check on the
cassette), and :func:`no_mutating_tool_reexecuted` proves the safety claim on a
specific recording - under the given policy, no tool the recording did not mark
read-only would be re-run live.
"""

import re
from collections.abc import Callable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from reenact.evals._text import clip, extract_text
from reenact.replay import Player, ReplayMode, ReplayPolicy
from reenact.schema import LLMCallEvent, SideEffect, ToolCallEvent, Trajectory


class CriterionLevel(StrEnum):
    """Whether a failed or regressed check blocks the gate, or only warns.

    Hard assertions are always ``BLOCKING`` (the default). A structured criterion
    may be marked ``ADVISORY`` so a shaky quality signal surfaces to a human -
    reported in the diff and the PR comment - without ever flaking the merge gate;
    calibration (a later rung) is what promotes a criterion to blocking or demotes
    it to advisory.
    """

    BLOCKING = "blocking"
    ADVISORY = "advisory"


class RunView:
    """A read-only view over a recorded trajectory, for checks to read.

    Wraps a :class:`Trajectory` and exposes the pieces an assertion cares about -
    the final answer text, the tool calls, the LLM calls - without leaking the raw
    event union. Replay-based checks reach the whole trajectory via
    :attr:`trajectory` and build their own :class:`Player`.
    """

    def __init__(self, trajectory: Trajectory) -> None:
        self.trajectory = trajectory

    @property
    def llm_calls(self) -> list[LLMCallEvent]:
        return [e for e in self.trajectory.events if isinstance(e, LLMCallEvent)]

    @property
    def tool_calls(self) -> list[ToolCallEvent]:
        return [e for e in self.trajectory.events if isinstance(e, ToolCallEvent)]

    @property
    def final_answer(self) -> str:
        """Text of the last LLM call that produced any - ``""`` if none did."""
        for event in reversed(self.trajectory.events):
            if isinstance(event, LLMCallEvent):
                text = extract_text(event.response)
                if text:
                    return text
        return ""

    def tool_calls_named(self, name: str) -> list[ToolCallEvent]:
        return [e for e in self.tool_calls if e.name == name]


class CheckResult(BaseModel):
    """The outcome of one check: did it pass, and why.

    ``score`` is unset for a boolean assertion; the LLM judge (a later rung) fills
    it with a graded value. ``level`` decides whether a regression on this check
    blocks the gate (the default) or is only an advisory warning - hard assertions
    leave it ``BLOCKING``; a structured criterion may set it ``ADVISORY``.
    Serializable, so a CI baseline can store it and diff.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    message: str = ""
    score: float | None = None
    level: CriterionLevel = CriterionLevel.BLOCKING


# A check is a plain function from a run view to its result.
type Check = Callable[[RunView], CheckResult]


def answer_contains(needle: str, *, case_sensitive: bool = False) -> Check:
    """Pass iff the final answer contains ``needle`` (case-insensitive by default)."""
    label = f"answer_contains({needle!r})"

    def check(view: RunView) -> CheckResult:
        answer = view.final_answer
        haystack = answer if case_sensitive else answer.casefold()
        target = needle if case_sensitive else needle.casefold()
        passed = target in haystack
        message = (
            f"answer contains {needle!r}"
            if passed
            else f"expected {needle!r} in answer, got {clip(answer)!r}"
        )
        return CheckResult(name=label, passed=passed, message=message)

    return check


def answer_matches(pattern: str, *, flags: int = 0) -> Check:
    """Pass iff ``pattern`` (a regex) is found anywhere in the final answer."""
    label = f"answer_matches({pattern!r})"
    compiled = re.compile(pattern, flags)

    def check(view: RunView) -> CheckResult:
        answer = view.final_answer
        passed = compiled.search(answer) is not None
        message = (
            f"answer matches /{pattern}/"
            if passed
            else f"expected a match for /{pattern}/, got {clip(answer)!r}"
        )
        return CheckResult(name=label, passed=passed, message=message)

    return check


def called_tool(name: str) -> Check:
    """Pass iff a tool named ``name`` was called at least once."""
    label = f"called_tool({name!r})"

    def check(view: RunView) -> CheckResult:
        count = len(view.tool_calls_named(name))
        passed = count > 0
        message = (
            f"{name!r} was called {count} time(s)"
            if passed
            else f"expected a call to {name!r}, but it was never called"
        )
        return CheckResult(name=label, passed=passed, message=message)

    return check


def did_not_call_tool(name: str) -> Check:
    """Pass iff a tool named ``name`` was never called - e.g. a read-only run that
    must not post a reply."""
    label = f"did_not_call_tool({name!r})"

    def check(view: RunView) -> CheckResult:
        count = len(view.tool_calls_named(name))
        passed = count == 0
        message = (
            f"{name!r} was not called"
            if passed
            else f"expected no call to {name!r}, but it was called {count} time(s)"
        )
        return CheckResult(name=label, passed=passed, message=message)

    return check


def tool_call_count(name: str, expected: int) -> Check:
    """Pass iff ``name`` was called exactly ``expected`` times."""
    label = f"tool_call_count({name!r}, {expected})"

    def check(view: RunView) -> CheckResult:
        count = len(view.tool_calls_named(name))
        passed = count == expected
        message = (
            f"{name!r} called {count} time(s) as expected"
            if passed
            else f"expected {name!r} to be called {expected} time(s), got {count}"
        )
        return CheckResult(name=label, passed=passed, message=message)

    return check


def replays_clean() -> Check:
    """Pass iff the recording replays offline with no divergence.

    Feeds every recorded request back through a :class:`Player`; each must
    reproduce its recorded response and every tool call must match its window. A
    failure means the cassette is internally inconsistent (a hand-edited body, a
    stale hash) - the integrity check that ties a scenario to the replay engine.
    """
    label = "replays_clean"

    def check(view: RunView) -> CheckResult:
        player = Player(view.trajectory, mode=ReplayMode.LENIENT)
        for event in view.trajectory.events:
            if isinstance(event, LLMCallEvent):
                player.replay_llm_call(event.request)
            elif isinstance(event, ToolCallEvent):
                player.replay_tool_call(event.name, event.arguments)
        divergences = player.divergences
        passed = not divergences
        message = (
            "recording replays offline with no divergence"
            if passed
            else f"{len(divergences)} divergence(s) on replay: {divergences[0].message}"
        )
        return CheckResult(name=label, passed=passed, message=message)

    return check


def no_mutating_tool_reexecuted(policy: ReplayPolicy | None = None) -> Check:
    """Pass iff no tool the recording did not mark read-only would be re-run live.

    The replay guarantee is that mutating (and unknown) tools are substituted,
    never re-fired. This verifies it holds for *this* recording under ``policy``:
    any tool the policy would re-execute must be recorded ``READ_ONLY``. The
    default policy substitutes everything, so it passes; a policy override that
    opts a mutating tool into live re-execution is what makes it fail.
    """
    label = "no_mutating_tool_reexecuted"

    def check(view: RunView) -> CheckResult:
        active = policy if policy is not None else ReplayPolicy()
        offenders = sorted(
            {
                event.name
                for event in view.tool_calls
                if event.side_effect is not SideEffect.READ_ONLY
                and not active.should_substitute(event.name, event.side_effect)
            }
        )
        passed = not offenders
        message = (
            "every non-read-only tool call is substituted on replay"
            if passed
            else f"policy would re-execute non-read-only tool(s): {offenders}"
        )
        return CheckResult(name=label, passed=passed, message=message)

    return check
