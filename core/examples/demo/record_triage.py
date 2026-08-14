"""Record the issue-triage demo corpus against the live Anthropic API.

Run once with a key to (re)generate ``examples/demo/scenarios/``: the baseline
suite (the good agent), one break-me set per seeded regression (model swap,
prompt edit, tool-schema change), and a clean-PR set (a benign reword that must
not regress). The recorder redacts before writing, so the cassettes are safe to
commit and replay offline forever.

    export ANTHROPIC_API_KEY=sk-ant-...
    PYTHONPATH=examples python examples/demo/record_triage.py
"""

# LangGraph/LangChain ship incomplete type stubs; basic mode keeps their generics
# from tripping strict checking.
# pyright: basic

import os
from datetime import UTC, datetime
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

from demo.fixtures import ISSUES, issue_text
from demo.triage_agent import (
    TOOLS,
    TRIAGE_SYSTEM,
    post_reply,
    read_file,
    search_docs,
    triage_trajectory,
)
from reenact.store import save_cassette

SCENARIOS = Path(__file__).resolve().parent / "scenarios"
SONNET = "claude-sonnet-4-5"
HAIKU = "claude-haiku-4-5-20251001"

# Pinned so regenerating diffs only when recorded behavior changes.
_FIXED_TIME = datetime(2024, 1, 1, tzinfo=UTC)

# prompt-edit regression: a degraded prompt that drops the triage steps.
WEAK_SYSTEM = (
    "You help with support issues. Read the issue and write a one-sentence reply "
    "to the user."
)

# clean-PR change: a benign reword that should not change what the agent does.
CONCISE_SYSTEM = TRIAGE_SYSTEM + " Keep every reply to one sentence."


# tool-schema regression: label_issue renamed to set_label, so an agent told to
# apply a label calls the new name and `called_tool('label_issue')` fails.
@tool
def set_label(issue_id: str, label: str) -> str:
    """Apply a category label to an issue (mutating)."""
    return f"labeled issue #{issue_id} as {label}"


ALT_TOOLS = [search_docs, read_file, set_label, post_reply]


def _model(name: str, tools: list) -> object:
    return ChatAnthropic(model=name, max_tokens=1024, temperature=0).bind_tools(tools)


def _record_set(
    name: str,
    *,
    model_name: str = SONNET,
    tools: list = TOOLS,
    system: str = TRIAGE_SYSTEM,
) -> None:
    out = SCENARIOS / name
    out.mkdir(parents=True, exist_ok=True)
    model = _model(model_name, tools)
    for issue_id in ISSUES:
        trajectory = triage_trajectory(
            model,
            issue_text(issue_id),
            tools=tools,
            system=system,
            name=f"triage-{issue_id}",
        )
        trajectory.id = f"demo-{name}-{issue_id}"
        trajectory.created_at = _FIXED_TIME
        save_cassette(trajectory, out / f"issue-{issue_id}.json")
        tool_calls = sum(1 for e in trajectory.events if e.type == "tool_call")
        print(
            f"  {name}/issue-{issue_id}: "
            f"{len(trajectory.events)} events, {tool_calls} tools"
        )


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set.")
    print("baseline (good agent, sonnet):")
    _record_set("baseline")
    print("break-me: model swap (haiku):")
    _record_set("model-swap", model_name=HAIKU)
    print("break-me: prompt edit (degraded prompt):")
    _record_set("prompt-edit", system=WEAK_SYSTEM)
    print("break-me: tool-schema change (label_issue -> set_label):")
    _record_set("tool-schema", tools=ALT_TOOLS)
    print("clean-pr (benign reword):")
    _record_set("clean-pr", system=CONCISE_SYSTEM)
    print(f"wrote scenarios under {SCENARIOS}")


if __name__ == "__main__":
    main()
