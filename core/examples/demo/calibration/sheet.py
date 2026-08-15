"""Generate the calibration labelling sheets from the demo runs + stress probes.

Two artifacts over the same items, so a human and an LLM judge (e.g. Fable 5) can
each answer the *identical* yes/no criteria on the *identical* runs, blind:

- ``human_sheet.md``  - readable transcripts + a fill-in ``= `` after each item.
- ``evaluator_prompt.txt`` - the same runs and criteria, asking for a JSON array.

Plus ``items.json`` - the id -> (trajectory, group, criterion) map the ingest step
(a later rung) uses to turn the two filled sheets into ``Label``s and calibrate.

The grid is the 15 real recorded runs (``group=real``, the headline) plus the 3
synthetic stress probes (``group=stress``), so the calibration has guaranteed hard
cases. Which run is which is recorded only in ``items.json`` - never shown on a
sheet, so a rater cannot tell the stress cases apart and label them differently.
"""

import ast
import json
from dataclasses import dataclass
from pathlib import Path

from demo.calibration.probes import probes
from demo.suites import QUALITY_CRITERIA
from reenact.evals import RunView
from reenact.evals._text import extract_text
from reenact.schema import LLMCallEvent, ToolCallEvent, Trajectory
from reenact.store import load_cassette

SETS = ("baseline", "model-swap", "prompt-edit", "tool-schema", "clean-pr")
ISSUES = ("42", "57", "63")
_SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"


@dataclass(frozen=True)
class SheetTrajectory:
    """One run on the sheet: its id, its group (real/stress), and the trajectory."""

    id: str
    group: str
    trajectory: Trajectory


def collect_trajectories() -> list[SheetTrajectory]:
    """The 15 real recorded runs followed by the 3 synthetic stress probes."""
    out: list[SheetTrajectory] = []
    for set_name in SETS:
        for issue in ISSUES:
            path = _SCENARIOS / set_name / f"issue-{issue}.json"
            traj = load_cassette(path)
            out.append(SheetTrajectory(f"{set_name}/issue-{issue}", "real", traj))
    for probe in probes():
        out.append(SheetTrajectory(f"probe/{probe.name}", "stress", probe))
    return out


def item_id(trajectory_id: str, criterion_id: str) -> str:
    return f"{trajectory_id}::{criterion_id}"


def build_items(trajectories: list[SheetTrajectory]) -> list[dict[str, str]]:
    """Every (run, criterion) pair on the grid, tagged with its group."""
    return [
        {
            "id": item_id(st.id, criterion.id),
            "trajectory_id": st.id,
            "group": st.group,
            "criterion": criterion.id,
        }
        for st in trajectories
        for criterion in QUALITY_CRITERIA
    ]


def _issue_text(trajectory: Trajectory) -> str:
    """The user's issue, from the first human/user message of the first LLM call."""
    for event in trajectory.events:
        if isinstance(event, LLMCallEvent):
            messages = event.request.get("messages")
            if isinstance(messages, list):
                for message in messages:  # pyright: ignore[reportUnknownVariableType]
                    if isinstance(message, dict):
                        role = message.get("role") or message.get("type")
                        content = message.get("content")
                        if role in ("user", "human") and isinstance(content, str):
                            return content
            return ""
    return ""


def _unwrap_args(arguments: dict[str, object]) -> dict[str, object]:
    """Peel the nested ``{"input": "{'query': ...}"}`` shape the demo tools use."""
    inner = arguments.get("input")
    if isinstance(inner, str):
        try:
            parsed = ast.literal_eval(inner)
        except (ValueError, SyntaxError):
            return arguments
        if isinstance(parsed, dict):
            return parsed  # pyright: ignore[reportUnknownVariableType]
    return arguments


def _result_text(result: object) -> str:
    """The meaningful text of a tool result (its ``content``, or the value itself)."""
    if isinstance(result, dict):
        content = result.get("content")  # pyright: ignore[reportUnknownMemberType]
        if isinstance(content, str):
            return content
    if isinstance(result, str):
        return result
    return json.dumps(result, default=str)


def _describe_tool(
    name: str, arguments: dict[str, object], result: object
) -> list[str]:
    """One or two plain-language lines for a tool call - no JSON, no wrappers."""
    args = _unwrap_args(arguments)
    text = _result_text(result)
    if name == "search_docs":
        return [f'Searched the docs for "{args.get("query", "")}"', f"   Found: {text}"]
    if name == "read_file":
        return [f'Read the file "{args.get("path", "")}"', f"   Contents: {text}"]
    if name == "label_issue":
        return [f"Labeled the issue: {args.get('label', '')}"]
    if name == "post_reply":
        return ["Posted this reply:", f'   "{args.get("body", "")}"']
    return [f"Called {name} -> {text}"]


def humanize_trajectory(trajectory: Trajectory) -> str:
    """A plain, readable summary of a run for a person to judge.

    Same content as the raw transcript - the issue, each thing the agent said or
    did, and its final summary - but with the machinery stripped: no ``node:``
    lines, no JSON argument/result blobs (tool results are unwrapped to their
    actual text), no step indices. Presentation only; nothing is added or removed.
    """
    final = RunView(trajectory).final_answer.strip()
    issue = _issue_text(trajectory).strip()
    lines: list[str] = []
    if issue:
        lines += ["ISSUE:", issue, ""]
    lines.append("WHAT THE AGENT DID:")
    step = 1
    for event in trajectory.events:
        if isinstance(event, LLMCallEvent):
            text = extract_text(event.response).strip()
            if text and text != final:  # the final summary is shown once, below
                lines.append(f"{step}. {text}")
                step += 1
        elif isinstance(event, ToolCallEvent):
            described = _describe_tool(event.name, event.arguments, event.result)
            lines.append(f"{step}. {described[0]}")
            lines.extend(described[1:])
            step += 1
    if final:
        lines += ["", "THE AGENT'S FINAL SUMMARY:", final]
    return "\n".join(lines)


def render_human_sheet(trajectories: list[SheetTrajectory]) -> str:
    """A blind, readable checklist: transcripts + a ``= `` to fill yes/no after."""
    lines = [
        "# Calibration labelling sheet (human rater)",
        "",
        "Read each run's transcript, then answer each criterion **yes** or **no** by",
        "writing your answer after the `=` on its line. Judge each run only on what",
        "its transcript shows. Label independently - do not look at anyone else's",
        "answers, including any model's.",
        "",
        f"{len(trajectories)} runs x {len(QUALITY_CRITERIA)} criteria.",
        "",
        "## Criteria",
    ]
    lines += [f"- **{c.id}** - {c.question}" for c in QUALITY_CRITERIA]
    lines.append("")
    for index, st in enumerate(trajectories, start=1):
        lines += [
            f"## Run {index} of {len(trajectories)} - `{st.id}`",
            "",
            "```",
            humanize_trajectory(st.trajectory),
            "```",
            "",
            "Answers (write yes or no after each `=`):",
        ]
        lines += [f"- `{item_id(st.id, c.id)}` = " for c in QUALITY_CRITERIA]
        lines.append("")
    return "\n".join(lines)


def render_evaluator_prompt(trajectories: list[SheetTrajectory]) -> str:
    """The same grid as a single prompt asking an LLM judge for a JSON array."""
    lines = [
        "You are a strict evaluator of an AI agent's issue-triage runs. For each run",
        "below you get a plain summary of what the agent did and a set of yes/no",
        "criteria. Answer every criterion for every run, citing the numbered step",
        "(and/or a short quote) that justifies your answer. Mark a criterion passed",
        "only when a concrete step supports it.",
        "",
        "Output ONLY a JSON array, one object per item id listed, of the form:",
        '[{"id": "<item id>", "passed": true|false, "evidence": "<step no./quote>"}]',
        "",
        "Criteria:",
    ]
    lines += [f"- {c.id}: {c.question}" for c in QUALITY_CRITERIA]
    lines.append("")
    for st in trajectories:
        lines += [
            f"=== Run: {st.id} ===",
            humanize_trajectory(st.trajectory),
            "Items to answer for this run:",
        ]
        lines += [f"- {item_id(st.id, c.id)}" for c in QUALITY_CRITERIA]
        lines.append("")
    return "\n".join(lines)


def write_sheets(out_dir: str | Path) -> dict[str, Path]:
    """Write the human sheet, the evaluator prompt, and items.json to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trajectories = collect_trajectories()
    paths = {
        "human": out / "human_sheet.md",
        "evaluator": out / "evaluator_prompt.txt",
        "items": out / "items.json",
    }
    paths["human"].write_text(render_human_sheet(trajectories), encoding="utf-8")
    paths["evaluator"].write_text(
        render_evaluator_prompt(trajectories), encoding="utf-8"
    )
    manifest = {
        "criteria": [{"id": c.id, "question": c.question} for c in QUALITY_CRITERIA],
        "items": build_items(trajectories),
    }
    paths["items"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return paths


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "sheets"
    for name, path in write_sheets(target).items():
        print(f"wrote {name}: {path}")
