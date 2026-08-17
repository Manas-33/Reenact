"""Propose an eval suite from a recording: the `reenact suggest` deriver.

Because reenact captured the whole trajectory, a suite can be *derived* from what
an agent actually did rather than written from a blank page. This module walks a
recorded trajectory and proposes checks - a `called_tool` per tool it used, the
`no_mutating_tool_reexecuted` safety net when it touched a mutating tool, and an
`answer_contains` keyword guessed from the run - then renders them as a candidate
`suite.toml` the author reviews and prunes. Everything is a *suggestion*: nothing
here gates, so a poor guess costs one line to delete, never a bad merge.

The structural derivation is deterministic and needs no model - pure trajectory
inspection, offline and free. The keyword guess is the one heuristic: it proposes a
word that appears in *both* the first user message and the final answer (a topical
anchor, not incidental phrasing), preferring a distinctive code like `429` and
otherwise the longest such word, and abstains entirely when nothing overlaps rather
than guessing wrong. Rendered output round-trips through
:func:`~reenact.evals.suite.load_suite`.

An optional second layer (:func:`suggest_criteria`) proposes evidence-based quality
*criteria* with the author's own model client - the one part that costs a call, on
the author's key. It is fail-open: a garbled reply yields no criteria, and the caller
skips it entirely without a client, so the structural suite is always produced.
Proposed criteria render commented-out (accepting one is uncommenting it).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, ValidationError

from reenact.evals._text import response_text
from reenact.evals.check import CriterionLevel, RunView
from reenact.evals.judge import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, render_trajectory
from reenact.evals.structured import Criterion
from reenact.schema import LLMCallEvent, SideEffect, ToolCallEvent, Trajectory


def _no_params() -> dict[str, str | int]:
    return {}


@dataclass(frozen=True)
class CheckSuggestion:
    """One proposed check, before it is rendered into a suite table.

    ``type`` and ``params`` mirror a ``[[scenario.check]]`` table; ``rationale`` is
    the one-line comment rendered above the block (shared rationales are printed
    once for a run of checks). ``active`` is ``False`` for a suggestion rendered
    commented-out - a tighter alternative the author can opt into.
    """

    type: str
    rationale: str
    params: dict[str, str | int] = field(default_factory=_no_params)
    active: bool = True


def _no_checks() -> list[CheckSuggestion]:
    return []


def _no_criteria() -> list[Criterion]:
    return []


@dataclass(frozen=True)
class ScenarioSuggestion:
    """One scenario's worth of suggestions, ready to render into a suite table.

    ``cassette`` is the path string written into the table (resolved by the caller
    relative to where the suite will live). ``checks`` are the structural
    suggestions; ``criteria`` are optional AI-proposed quality criteria, rendered
    commented-out.
    """

    name: str
    cassette: str
    checks: list[CheckSuggestion] = field(default_factory=_no_checks)
    criteria: list[Criterion] = field(default_factory=_no_criteria)


# --- keyword heuristic -------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+")
# Reference ids like "#42" are bookkeeping, not topic: an "Issue #42" prompt and an
# "triaged issue #42" answer share "42", which would otherwise win the digit
# preference. Stripped before tokenizing; a bare code like "429" (no "#") survives.
_REF = re.compile(r"#\d+")
_KEYWORD_MIN_LEN = 4

# Generic filler words that carry no topic, dropped before the prompt/answer
# overlap is taken so a shared "the"/"error"/"help" is never proposed as a keyword.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "there",
        "here",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "not",
        "no",
        "yes",
        "so",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "do",
        "does",
        "did",
        "done",
        "doesn",
        "dont",
        "don",
        "didn",
        "have",
        "has",
        "had",
        "having",
        "to",
        "of",
        "in",
        "on",
        "for",
        "from",
        "with",
        "without",
        "at",
        "by",
        "as",
        "into",
        "onto",
        "over",
        "under",
        "about",
        "after",
        "before",
        "out",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "us",
        "them",
        "my",
        "your",
        "his",
        "her",
        "its",
        "our",
        "their",
        "mine",
        "yours",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
        "can",
        "could",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "get",
        "getting",
        "got",
        "please",
        "help",
        "need",
        "want",
        "use",
        "using",
        "work",
        "works",
        "working",
        "issue",
        "error",
        "errors",
        "problem",
    }
)


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
                text = cast(dict[str, Any], block).get("text")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(parts)
    return ""


def _first_user_text(trajectory: Trajectory) -> str:
    """Text of the first user/human message in the run's first LLM request."""
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
            if role in ("user", "human"):
                return _content_text(mapping.get("content"))
        return ""
    return ""


def _has_digit(token: str) -> bool:
    return any(character.isdigit() for character in token)


def _keyword_suggestion(trajectory: Trajectory) -> str | None:
    """A topical keyword shared by the prompt and the answer, or ``None``.

    Candidates are words present in both the first user message and the final
    answer, minus stopwords, keeping a token only if it is at least
    ``_KEYWORD_MIN_LEN`` characters or contains a digit (so a short code like
    ``429`` survives). A digit-bearing token wins over a plain word, then the
    longest; ties keep first appearance in the answer. Returns ``None`` when
    nothing overlaps - abstaining beats a wrong guess.
    """
    prompt = _REF.sub(" ", _first_user_text(trajectory))
    answer = _REF.sub(" ", RunView(trajectory).final_answer)
    if not prompt or not answer:
        return None
    prompt_tokens = set(_TOKEN.findall(prompt.lower()))
    if not prompt_tokens:
        return None
    candidates: list[str] = []
    for token in _TOKEN.findall(answer.lower()):
        if token in candidates or token in _STOPWORDS or token not in prompt_tokens:
            continue
        if len(token) >= _KEYWORD_MIN_LEN or _has_digit(token):
            candidates.append(token)
    if not candidates:
        return None
    # Stable sort: digit-bearing first, then longest; ties keep answer order.
    candidates.sort(key=lambda token: (0 if _has_digit(token) else 1, -len(token)))
    return candidates[0]


# --- structural deriver ------------------------------------------------------


def suggest_structural(trajectory: Trajectory) -> list[CheckSuggestion]:
    """Propose checks from what the recording did, deterministically (no model).

    Emits a `called_tool` per distinct tool in first-appearance order, the
    `no_mutating_tool_reexecuted` safety check if any tool was not read-only, an
    `answer_contains` keyword when a topical anchor exists, and - commented, as a
    tighter alternative - a `tool_call_count` for the first tool.
    """
    tool_calls = [e for e in trajectory.events if isinstance(e, ToolCallEvent)]

    counts: dict[str, int] = {}
    for event in tool_calls:
        counts[event.name] = counts.get(event.name, 0) + 1

    suggestions: list[CheckSuggestion] = [
        CheckSuggestion(
            type="called_tool",
            rationale="the agent called each of these tools during the run",
            params={"name": name},
        )
        for name in counts
    ]

    if any(event.side_effect is not SideEffect.READ_ONLY for event in tool_calls):
        suggestions.append(
            CheckSuggestion(
                type="no_mutating_tool_reexecuted",
                rationale="a mutating tool ran - replay must never re-fire it",
            )
        )

    keyword = _keyword_suggestion(trajectory)
    if keyword is not None:
        suggestions.append(
            CheckSuggestion(
                type="answer_contains",
                rationale="guessed keyword - keep only if it is a stable anchor",
                params={"value": keyword},
            )
        )

    if counts:
        first_tool = next(iter(counts))
        suggestions.append(
            CheckSuggestion(
                type="tool_call_count",
                rationale="optional: tighten a called_tool to an exact count",
                params={"name": first_tool, "count": counts[first_tool]},
                active=False,
            )
        )

    return suggestions


# --- optional AI quality-criteria layer --------------------------------------

_MAX_CRITERIA = 4

SUGGEST_CRITERIA_SYSTEM = (
    "You help an engineer set up regression tests for an AI agent. Given ONE "
    "recorded run - its task, the ordered steps it took, and its final answer - "
    "propose a short list of yes/no quality criteria a good run must satisfy, the "
    "kind a reviewer checks to catch a regression. Each must be answerable from the "
    "trajectory with evidence (a cited step), phrased as a yes/no question. Favour "
    "groundedness (is the answer supported by what the agent retrieved?), the "
    "correctness of key decisions (right category, label, or action), and task "
    "completion. Do NOT restate which tools were called - that is covered by "
    "separate assertions. Respond with ONLY a JSON array of 2-4 objects, each of "
    'the form {"id": "<snake_case>", "question": "<yes/no question>"}. Nothing else.'
)


class _ProposedCriterion(BaseModel):
    """One criterion parsed from the proposer's reply (extra keys tolerated)."""

    model_config = ConfigDict(extra="ignore")

    id: str
    question: str


def _parse_criteria(text: str) -> list[Criterion]:
    """Parse the proposer's reply (first ``[`` .. last ``]``) into criteria.

    Defensive like the judge and structured evaluator: a reply with no array, or an
    item that fails validation or is empty/duplicate, contributes nothing rather than
    raising - so a garbled proposal simply yields no criteria (fail-open).
    """
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    criteria: list[Criterion] = []
    seen: set[str] = set()
    for item in cast(list[Any], data):
        if not isinstance(item, dict):
            continue
        try:
            proposed = _ProposedCriterion.model_validate(item)
        except ValidationError:
            continue
        cid = proposed.id.strip()
        question = proposed.question.strip()
        if not cid or not question or cid in seen:
            continue
        seen.add(cid)
        criteria.append(Criterion(id=cid, question=question))
    return criteria[:_MAX_CRITERIA]


def suggest_criteria(
    client: Any,
    trajectory: Trajectory,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
) -> list[Criterion]:
    """Propose evidence-based quality criteria from a run, using ``client``.

    Renders the trajectory and asks the model for a small set of yes/no criteria.
    The client is duck-typed (the same rule as the judge): it calls
    ``client.messages.create(...)`` and reads the reply, importing no SDK. A reply
    that does not parse yields ``[]`` - the caller renders them commented, so nothing
    here ever gates. Raises only if the client itself does; the CLI treats that as
    best-effort and falls back to structural checks.
    """
    prompt = f"{render_trajectory(trajectory)}\n\nPropose the quality criteria."
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=SUGGEST_CRITERIA_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_criteria(response_text(response))


# --- TOML rendering ----------------------------------------------------------


def _toml_str(value: str) -> str:
    """Escape a string for a double-quoted TOML value."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _check_body(suggestion: CheckSuggestion) -> list[str]:
    """The raw (un-indented, un-commented) TOML lines for one check table."""
    lines = ["[[scenario.check]]", f'type = "{_toml_str(suggestion.type)}"']
    for key, value in suggestion.params.items():
        if isinstance(value, bool):  # bool is an int subclass - handle first
            lines.append(f"{key} = {str(value).lower()}")
        elif isinstance(value, int):
            lines.append(f"{key} = {value}")
        else:
            lines.append(f'{key} = "{_toml_str(value)}"')
    return lines


def _criterion_body(criterion: Criterion) -> list[str]:
    """The raw (un-indented, un-commented) TOML lines for one criterion table."""
    lines = [
        "[[scenario.criterion]]",
        f'id = "{_toml_str(criterion.id)}"',
        f'question = "{_toml_str(criterion.question)}"',
    ]
    if criterion.level is not CriterionLevel.BLOCKING:
        lines.append(f'level = "{criterion.level.value}"')
    return lines


# Shown when no criteria were proposed (no client/key): an illustrative example.
_CRITERION_EXAMPLE = [
    "",
    "# Quality criteria (evidence-backed, need your API key at eval time) sit beside",
    "# the checks as [[scenario.criterion]] tables. Review before trusting; uncomment",
    "# to enable. Each is a yes/no question answered with a citation to a step.",
    "#",
    "#   [[scenario.criterion]]",
    '#   id = "reply_grounded"',
    '#   question = "Is the reply grounded in what the agent retrieved, not invented?"',
]


def _scenario_block(scenario: ScenarioSuggestion) -> list[str]:
    """The `[[scenario]]` table for one scenario: header, cassette, checks, criteria.

    Checks render active (or commented for a tighter alternative); any proposed
    criteria render commented-out, so accepting one is just uncommenting it.
    """
    out = [
        "[[scenario]]",
        f'name = "{_toml_str(scenario.name)}"',
        f'cassette = "{_toml_str(scenario.cassette)}"',
    ]
    previous_rationale: str | None = None
    for suggestion in scenario.checks:
        out.append("")
        if suggestion.rationale and suggestion.rationale != previous_rationale:
            out.append(f"  # {suggestion.rationale}")
        previous_rationale = suggestion.rationale
        prefix = "  " if suggestion.active else "  # "
        out.extend(f"{prefix}{line}" for line in _check_body(suggestion))
    if scenario.criteria:
        out.append("")
        out.append(
            "  # quality criteria proposed from the transcript - need your key; "
            "uncomment to use:"
        )
        for criterion in scenario.criteria:
            out.append("")
            out.extend(f"  # {line}" for line in _criterion_body(criterion))
    return out


def render_suite_toml(scenarios: Iterable[ScenarioSuggestion]) -> str:
    """Render one or more scenarios into a candidate `suite.toml`.

    A header comment, then one `[[scenario]]` table per scenario. When no scenario
    proposes a criterion, a short illustrative placeholder is shown once. The active
    tables load cleanly through :func:`~reenact.evals.suite.load_suite`; commented
    lines are ignored, so the suggested suite loads with no client.
    """
    entries = list(scenarios)
    out: list[str] = [
        "# Suggested by `reenact suggest`.",
        "# Derived from what the agent did - keep what applies, delete the rest.",
    ]
    for scenario in entries:
        out.append("")
        out.extend(_scenario_block(scenario))
    if not any(scenario.criteria for scenario in entries):
        out.extend(_CRITERION_EXAMPLE)
    return "\n".join(out) + "\n"
