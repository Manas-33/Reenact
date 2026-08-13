"""A LangGraph issue-triage agent - the demo's breakable star.

A real think -> act loop: the model searches the docs, applies a label, posts a
reply, and summarizes. Four tools, two read-only (``search_docs`` / ``read_file``)
and two mutating (``label_issue`` / ``post_reply``). Recorded through the reenact
LangChain callback handler, so a run captures the LLM calls, the tool calls, and
the LangGraph node boundaries at once.

``build_triage_graph`` takes any object the agent node can ``.invoke`` - a real
tool-bound ``ChatAnthropic`` when recording, a scripted fake model when testing
offline - so the same graph records real trajectories and runs in CI with no key.
"""

# LangGraph/LangChain ship incomplete type stubs; run this demo module in basic
# mode so their partially-unknown generics don't trip strict checking.
# pyright: basic

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from demo.fixtures import DOCS
from reenact.record.langchain import ReenactCallbackHandler
from reenact.schema import SideEffect, ToolCallEvent, Trajectory


@tool
def search_docs(query: str) -> str:
    """Search the product docs and return the most relevant document's text."""
    words = set(query.lower().split())
    best = max(DOCS.items(), key=lambda item: len(words & set(item[1].lower().split())))
    return best[1]


@tool
def read_file(path: str) -> str:
    """Read a documentation file by name, e.g. 'auth.md'."""
    return DOCS.get(path, f"file not found: {path}")


@tool
def label_issue(issue_id: str, label: str) -> str:
    """Apply a category label to an issue (mutating)."""
    return f"labeled issue #{issue_id} as {label}"


@tool
def post_reply(issue_id: str, body: str) -> str:
    """Post a public reply on an issue (mutating)."""
    return f"posted reply to issue #{issue_id}"


TOOLS = [search_docs, read_file, label_issue, post_reply]
READ_ONLY_TOOLS = frozenset({"search_docs", "read_file"})
MUTATING_TOOLS = frozenset({"label_issue", "post_reply"})

TRIAGE_SYSTEM = (
    "You are an issue-triage assistant for a software product. For the issue you "
    "are given: search the docs for relevant guidance, apply exactly one category "
    "label (one of: bug, billing, api), post a short helpful reply to the user, "
    "and finish with a one-sentence summary of what you did."
)


class _State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_triage_graph(model: Any) -> Any:
    """Compile the triage graph around ``model`` (already tool-bound, or a fake)."""

    def agent(state: _State) -> dict[str, Any]:
        return {"messages": [model.invoke(state["messages"])]}

    def route(state: _State) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    builder = StateGraph(_State)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile()


def label_side_effects(trajectory: Trajectory) -> Trajectory:
    """Set each recorded tool call's side-effect class by name.

    The blanket callback handler records every tool as UNKNOWN; the demo knows
    which tools mutate, so this stamps the real class onto the committed cassette -
    what makes ``no_mutating_tool_reexecuted`` meaningful on these recordings.
    """
    for event in trajectory.events:
        if isinstance(event, ToolCallEvent):
            if event.name in MUTATING_TOOLS:
                event.side_effect = SideEffect.MUTATING
            elif event.name in READ_ONLY_TOOLS:
                event.side_effect = SideEffect.READ_ONLY
    return trajectory


def triage_trajectory(model: Any, issue: str, *, name: str | None = None) -> Trajectory:
    """Run the triage agent on ``issue`` and return the recorded trajectory."""
    handler = ReenactCallbackHandler()
    graph = build_triage_graph(model)
    graph.invoke(
        {
            "messages": [
                SystemMessage(content=TRIAGE_SYSTEM),
                HumanMessage(content=issue),
            ]
        },
        config={"callbacks": [handler]},
    )
    trajectory = handler.recorder.trajectory
    if name is not None:
        trajectory.name = name
    return label_side_effects(trajectory)
