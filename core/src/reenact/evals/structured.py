"""Structured trajectory evaluator: fuzzy quality as evidence-backed soft assertions.

Where the scalar judge (:mod:`reenact.evals.judge`) collapses a whole run to one
0-1 float - noisy, and hard to gate on without an arbitrary threshold - the
structured evaluator asks the model a set of yes/no *criteria*, each answered with
a citation to the trajectory step that justifies it. Every criterion becomes an
ordinary :class:`~reenact.evals.check.CheckResult`, so it flows through the same
runner, baseline, and diff as a hard assertion: a regression is simply a criterion
that flipped pass->fail, with no weighted sum and no tolerance to tune.

A passing verdict must cite a real step of the trajectory. A "pass" with no
evidence - or one citing a step that does not exist - is downgraded to fail, so a
hallucinated pass cannot slip a bad run through the gate (the same fail-closed
posture as the judge's unparseable-reply rule). The downgrade only ever tightens a
verdict (pass->fail), never loosens one, so it can never mask a real regression.

The evaluator answers every criterion in a single model call and memoizes it per
run, so N criteria cost one judge call, not N. The client is duck-typed, the same
rule as the judge and the SDK recorders: it calls ``client.messages.create(...)``
and reads the reply, importing no SDK. A deterministic stub drives the tests, so
the mechanism is green with no key; calibrating the criteria against human labels
(the per-criterion agreement number) is a later rung.
"""

import json
import re
from collections.abc import Iterable
from typing import Any, cast
from weakref import WeakKeyDictionary

from pydantic import BaseModel, ConfigDict, ValidationError

from reenact.evals._text import clip, response_text
from reenact.evals.check import Check, CheckResult, CriterionLevel, RunView
from reenact.evals.judge import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, render_trajectory

STRUCTURED_SYSTEM = (
    "You are a strict evaluator of an AI agent's multi-step trajectory. You are "
    "given the agent's task, the full ordered sequence of steps it took (each "
    "prefixed with its step index in brackets, e.g. [2]), its final answer, and a "
    "list of yes/no criteria. Answer EVERY criterion independently. For each, "
    "decide whether the trajectory satisfies it, and cite the specific step that "
    "justifies your answer by its bracketed index (e.g. [2]); quote from that step "
    "if it helps. Mark a criterion 'passed' only when a concrete step supports it - "
    "if nothing in the trajectory supports it, mark it failed. Respond with ONLY a "
    "JSON array, one object per criterion, each of the form "
    '{"id": "<criterion id>", "passed": <true|false>, "evidence": "<step index '
    'and/or quote>", "reasoning": "<one sentence>"}. Do not output anything else.'
)

_STEP_REF = re.compile(r"\[(\d+)\]")


class Criterion(BaseModel):
    """A yes/no quality question about a trajectory, answered with evidence.

    ``id`` is the stable key a check is named for (so it matches across a baseline
    diff); ``question`` is the yes/no prompt the evaluator answers. ``level`` marks
    whether a regression on this criterion blocks the gate (``BLOCKING``, the
    default) or is only a reported warning (``ADVISORY``).
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    level: CriterionLevel = CriterionLevel.BLOCKING


class CriterionVerdict(BaseModel):
    """One criterion's answer parsed from the evaluator's reply."""

    model_config = ConfigDict(extra="ignore")

    id: str
    passed: bool
    evidence: str = ""
    reasoning: str = ""


# A task-general criterion: unique to full-trajectory capture, and meaningful for
# any agent that retrieves before it answers. Answered by the same evaluator now;
# a deterministic NLI/embedding backend is a possible follow-up.
FAITHFULNESS = Criterion(
    id="faithful",
    question=(
        "Is the agent's final answer entailed by the evidence it actually "
        "retrieved during the run (tool results, retrieved documents) - i.e. not "
        "invented, and not contradicted by what the tools returned?"
    ),
)


def _cited_steps(evidence: str) -> set[int]:
    """Bracketed step indices referenced in an evidence string, e.g. ``[2]``."""
    return {int(match) for match in _STEP_REF.findall(evidence)}


def _grounded(verdict: CriterionVerdict, valid_seqs: set[int]) -> bool:
    """Whether a passing verdict cites a real step of the trajectory.

    A pass makes a positive claim, so it must point at a concrete step; evidence
    that cites nothing, or cites a step index the trajectory does not contain, is
    not grounded and downgrades the pass to a fail.
    """
    return bool(_cited_steps(verdict.evidence) & valid_seqs)


def _extract_json_array(text: str) -> list[Any]:
    """Take the first ``[`` through the last ``]`` and parse it as a JSON array.

    Mirrors the judge's defensive parse so a reply wrapped in prose or a code fence
    still yields the array. Raises ``ValueError`` if there is no array or it does
    not parse to a list.
    """
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end <= start:
        raise ValueError("no JSON array in evaluator reply")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, list):
        raise ValueError("evaluator reply is not a JSON array")
    return cast(list[Any], data)


def _parse_verdicts(text: str) -> dict[str, CriterionVerdict]:
    """Parse the evaluator's reply into verdicts keyed by criterion id.

    A reply with no array, or an item that fails validation, contributes no
    verdict; the caller reports any criterion left without one as a fail (so a
    garbled evaluation blocks a merge rather than silently passing).
    """
    try:
        items = _extract_json_array(text)
    except ValueError:
        return {}
    verdicts: dict[str, CriterionVerdict] = {}
    for item in items:
        if isinstance(item, dict):
            try:
                verdict = CriterionVerdict.model_validate(item)
            except ValidationError:
                continue
            verdicts[verdict.id] = verdict
    return verdicts


def _verdict_to_result(
    label: str,
    criterion: Criterion,
    verdict: CriterionVerdict | None,
    valid_seqs: set[int],
) -> CheckResult:
    """Turn one criterion's verdict into a soft assertion (a ``CheckResult``).

    The criterion's ``level`` rides onto the result, so an advisory criterion's
    pass->fail is reported by the gate but never blocks it.
    """
    if verdict is None:
        return CheckResult(
            name=label,
            passed=False,
            level=criterion.level,
            message=f"no verdict returned for criterion {criterion.id!r}",
        )
    if verdict.passed and not _grounded(verdict, valid_seqs):
        return CheckResult(
            name=label,
            passed=False,
            level=criterion.level,
            message=(
                "claimed pass without citing a real trajectory step "
                f"(evidence: {clip(verdict.evidence)!r}) - downgraded to fail"
            ),
        )
    detail = (
        f"evidence {clip(verdict.evidence)!r}" if verdict.evidence else "no evidence"
    )
    verb = "satisfied" if verdict.passed else "not satisfied"
    reason = f": {verdict.reasoning}" if verdict.reasoning else ""
    return CheckResult(
        name=label,
        passed=verdict.passed,
        level=criterion.level,
        message=f"{criterion.id} {verb} ({detail}){reason}",
    )


class StructuredEvaluator:
    """Evaluate a set of criteria over a run in one model call, as soft assertions.

    :meth:`evaluate` returns the raw per-criterion verdicts (``{id: verdict}``);
    :meth:`checks` returns one :data:`~reenact.evals.check.Check` per criterion for
    a scenario. The single evaluation is memoized per :class:`RunView`, so however
    many criterion-checks a scenario runs, the model is called once.
    """

    def __init__(
        self,
        client: Any,
        criteria: Iterable[Criterion],
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ) -> None:
        self.client = client
        self.criteria = list(criteria)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._cache: WeakKeyDictionary[RunView, dict[str, CriterionVerdict]] = (
            WeakKeyDictionary()
        )

    def evaluate(self, view: RunView) -> dict[str, CriterionVerdict]:
        """The per-criterion verdicts for ``view``, computed once and memoized."""
        cached = self._cache.get(view)
        if cached is None:
            cached = self._run(view)
            self._cache[view] = cached
        return cached

    def _run(self, view: RunView) -> dict[str, CriterionVerdict]:
        transcript = render_trajectory(view.trajectory)
        questions = "\n".join(f"- id={c.id}: {c.question}" for c in self.criteria)
        prompt = (
            f"{transcript}\n\nCriteria:\n{questions}\n\n"
            "Answer every criterion as specified."
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=STRUCTURED_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return _parse_verdicts(response_text(response))

    def checks(self) -> list[Check]:
        """One soft-assertion check per criterion, sharing a single evaluation."""
        return [self._check_for(criterion) for criterion in self.criteria]

    def _check_for(self, criterion: Criterion) -> Check:
        label = f"criterion:{criterion.id}"

        def check(view: RunView) -> CheckResult:
            verdicts = self.evaluate(view)
            valid_seqs = {event.seq for event in view.trajectory.events}
            return _verdict_to_result(
                label, criterion, verdicts.get(criterion.id), valid_seqs
            )

        return check


def structured_eval(
    client: Any,
    criteria: Iterable[Criterion],
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
) -> list[Check]:
    """One evidence-backed soft assertion per criterion, all sharing one call.

    A convenience over :class:`StructuredEvaluator`, parallel to the other check
    factories (``answer_contains``, ``judged``): drop the returned checks straight
    into a :class:`~reenact.evals.scenario.Scenario`. ``temperature`` defaults to 0
    so a gate scores as reproducibly as the model allows.
    """
    evaluator = StructuredEvaluator(
        client,
        criteria,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return evaluator.checks()
