"""The issue-triage demo agent, driven offline by a scripted fake model.

Proves the whole demo pipeline with no key: the real LangGraph graph runs, the
callback handler records the LLM/tool/node events, side-effects are labeled, and
the eval checks pass on the recording - including the final answer pulled from
LangChain's LLMResult shape. Recording the real corpus against the live model is
the next rung.
"""

# LangGraph/LangChain ship incomplete type stubs; basic mode keeps their generics
# from tripping strict checking of this test.
# pyright: basic

from typing import Any

import pytest

pytest.importorskip("langgraph")

from demo.fixtures import issue_text
from demo.triage_agent import (
    label_issue,
    search_docs,
    triage_trajectory,
)
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from reenact.evals import (
    RunView,
    Scenario,
    answer_contains,
    called_tool,
    no_mutating_tool_reexecuted,
    run_scenario,
)
from reenact.schema import SideEffect, ToolCallEvent


def _call(name: str, **args: Any) -> dict[str, Any]:
    return {"name": name, "args": args, "id": f"call_{name}"}


def _full_triage_model() -> Any:
    """A fake model scripted to search, label, reply, then summarize."""
    return GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        _call("search_docs", query="password reset link error")
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[_call("label_issue", issue_id="42", label="bug")],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        _call(
                            "post_reply",
                            issue_id="42",
                            body="Reset links expire in an hour.",
                        )
                    ],
                ),
                AIMessage(
                    content=(
                        "I labeled issue #42 as a bug and replied that the reset "
                        "link expires in an hour; reset via Settings > Security."
                    )
                ),
            ]
        )
    )


def test_tools_read_the_fixture() -> None:
    assert "Settings > Security" in search_docs.invoke({"query": "reset password"})
    assert "labeled issue #42" in label_issue.invoke({"issue_id": "42", "label": "bug"})


def test_triage_records_all_surfaces() -> None:
    trajectory = triage_trajectory(
        _full_triage_model(), issue_text("42"), name="triage-42"
    )
    kinds = {event.type for event in trajectory.events}
    assert {"llm_call", "tool_call", "graph_node"} <= kinds
    tool_names = [e.name for e in trajectory.events if isinstance(e, ToolCallEvent)]
    assert tool_names == ["search_docs", "label_issue", "post_reply"]


def test_side_effects_labeled_by_name() -> None:
    trajectory = triage_trajectory(_full_triage_model(), issue_text("42"))
    effects = {
        e.name: e.side_effect for e in trajectory.events if isinstance(e, ToolCallEvent)
    }
    assert effects["search_docs"] is SideEffect.READ_ONLY
    assert effects["label_issue"] is SideEffect.MUTATING
    assert effects["post_reply"] is SideEffect.MUTATING


def test_final_answer_extracted_from_langgraph_recording() -> None:
    trajectory = triage_trajectory(_full_triage_model(), issue_text("42"))
    assert "Settings > Security" in RunView(trajectory).final_answer


def test_eval_checks_pass_on_the_recording() -> None:
    trajectory = triage_trajectory(
        _full_triage_model(), issue_text("42"), name="triage-42"
    )
    scenario = Scenario(
        name="triage-42",
        trajectory=trajectory,
        checks=[
            called_tool("search_docs"),
            called_tool("label_issue"),
            called_tool("post_reply"),
            answer_contains("Settings"),
            no_mutating_tool_reexecuted(),
        ],
    )
    result = run_scenario(scenario)
    assert result.passed, result.failures
