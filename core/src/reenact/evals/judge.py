"""A trajectory-level LLM judge - a check that scores multi-step behavior.

Unlike a hard assertion, the judge reads the *whole* run - the task, every step
(LLM output, tool call, result), and the final answer - renders it into a
transcript, and asks a model to score how well it satisfies a rubric. It returns
a graded :class:`CheckResult` (``score`` in [0, 1], ``passed`` iff the score
clears a threshold), so a scenario can mix hard assertions with a soft judged
score.

The judge client is duck-typed, the same rule as the SDK recorders: the judge
calls ``client.messages.create(**request)`` and reads the response, importing no
SDK. A deterministic stub satisfies that shape in tests, so the mechanism is green
with no key; calibrating the judge against human labels (the agreement number) is
a later rung.
"""

import json
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from reenact.evals._text import clip, extract_text
from reenact.evals.check import Check, CheckResult, RunView
from reenact.schema import LLMCallEvent, ToolCallEvent, Trajectory

DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_THRESHOLD = 0.7
DEFAULT_MAX_TOKENS = 512

JUDGE_SYSTEM = (
    "You are a strict evaluator of an AI agent's multi-step trajectory. You are "
    "given the agent's task, the full ordered sequence of steps it took (LLM "
    "outputs, tool calls with their arguments and results), its final answer, and "
    "a rubric describing what a correct run looks like. Judge how well the whole "
    "trajectory satisfies the rubric: reward correct tool use, sound intermediate "
    "steps, and a final answer that follows from them; penalize wrong or missing "
    "tool calls, skipped steps, invented results, or a final answer the steps do "
    "not support. Respond with ONLY a JSON object of the form "
    '{"score": <number between 0 and 1>, "reasoning": "<one or two sentences>"}. '
    "Do not output anything else."
)


class JudgeVerdict(BaseModel):
    """The structured judgment parsed from the judge model's reply."""

    model_config = ConfigDict(extra="ignore")

    score: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


def _user_text(content: Any) -> str:
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


def _first_user_message(trajectory: Trajectory) -> str:
    """The first user message of the first LLM call - the run's task, if present."""
    for event in trajectory.events:
        if isinstance(event, LLMCallEvent):
            messages = event.request.get("messages")
            if isinstance(messages, list):
                for message in cast(list[Any], messages):
                    if isinstance(message, dict):
                        mapping = cast(dict[str, Any], message)
                        if mapping.get("role") == "user":
                            return _user_text(mapping.get("content"))
            return ""
    return ""


def _render_llm_output(response: dict[str, Any]) -> str:
    """An LLM step's output: its text, or a description of the tool calls it made."""
    text = extract_text(response)
    if text:
        return text
    content = response.get("content")
    calls: list[str] = []
    if isinstance(content, list):
        for block in cast(list[Any], content):
            if isinstance(block, dict):
                mapping = cast(dict[str, Any], block)
                if mapping.get("type") == "tool_use":
                    arguments = json.dumps(mapping.get("input", {}), sort_keys=True)
                    calls.append(f"calls {mapping.get('name')}({arguments})")
    return "; ".join(calls) if calls else "(no text output)"


def _render_result(result: Any) -> str:
    if isinstance(result, str):
        return clip(result)
    return clip(json.dumps(result, sort_keys=True, default=str))


def render_trajectory(trajectory: Trajectory) -> str:
    """Render a trajectory as a readable transcript for the judge to score.

    Shows the task, then each step in order - LLM outputs (text or the tool calls
    they request), tool calls with arguments and results, graph-node boundaries -
    and the final answer. This is the multi-step view the judge grades, not a
    single completion.
    """
    lines: list[str] = []
    task = _first_user_message(trajectory)
    if task:
        lines.append(f"Task: {task}")
        lines.append("")
    lines.append("Trajectory:")
    for event in trajectory.events:
        if isinstance(event, LLMCallEvent):
            lines.append(
                f"[{event.seq}] assistant: {_render_llm_output(event.response)}"
            )
        elif isinstance(event, ToolCallEvent):
            arguments = json.dumps(event.arguments, sort_keys=True)
            lines.append(
                f"[{event.seq}] tool {event.name}({arguments}) -> "
                f"{_render_result(event.result)}"
            )
        else:  # the only remaining event type is a graph-node boundary
            lines.append(f"[{event.seq}] node: {event.node}")
    lines.append("")
    lines.append(f"Final answer: {RunView(trajectory).final_answer}")
    return "\n".join(lines)


def _response_text(response: Any) -> str:
    """Pull the reply text from a judge client's response (object or dict)."""
    if hasattr(response, "model_dump"):
        body: Any = response.model_dump(mode="json")
        if isinstance(body, dict):
            return extract_text(cast(dict[str, Any], body))
    if isinstance(response, dict):
        return extract_text(cast(dict[str, Any], response))
    raise TypeError("judge client returned an unreadable response")


def _parse_verdict(text: str) -> JudgeVerdict:
    """Extract and validate the JSON verdict from the judge's reply text.

    Takes the first ``{`` through the last ``}`` so a reply wrapped in prose or a
    code fence still parses. Raises ``ValueError`` / ``ValidationError`` on a reply
    with no JSON object or an out-of-range score - handled by the caller.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in judge reply")
    data = json.loads(text[start : end + 1])
    return JudgeVerdict.model_validate(data)


def judged(
    client: Any,
    rubric: str,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    name: str | None = None,
) -> Check:
    """A check that scores the trajectory against ``rubric`` with an LLM judge.

    Passes iff the judged score is at least ``threshold``. A reply the judge
    cannot be parsed from fails the check (score unknown) rather than raising, so a
    garbled judgment blocks a merge instead of silently passing.
    """
    label = name or f"judge: {clip(rubric, 60)}"

    def check(view: RunView) -> CheckResult:
        transcript = render_trajectory(view.trajectory)
        prompt = (
            f"Rubric: {rubric}\n\n{transcript}\n\n"
            "Score how well the trajectory satisfies the rubric."
        )
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = _response_text(response)
        try:
            verdict = _parse_verdict(text)
        except (ValueError, ValidationError) as exc:
            return CheckResult(
                name=label,
                passed=False,
                score=None,
                message=f"could not parse judge reply ({exc}): {clip(text)!r}",
            )
        passed = verdict.score >= threshold
        relation = ">=" if passed else "<"
        summary = f"score {verdict.score:.2f} {relation} {threshold:.2f}"
        return CheckResult(
            name=label,
            passed=passed,
            score=verdict.score,
            message=f"{summary}: {verdict.reasoning}",
        )

    return check
