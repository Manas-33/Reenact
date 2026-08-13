"""The LangChain callback handler records LLM and tool calls into a trajectory."""

from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, Generation, LLMResult

from reenact.record.langchain import ReenactCallbackHandler
from reenact.schema import GraphNodeEvent, LLMCallEvent, ToolCallEvent


def _messages(text: str) -> list[list[BaseMessage]]:
    return [[HumanMessage(content=text)]]


def _llm_result(text: str) -> LLMResult:
    generations: list[list[Generation]] = [
        [ChatGeneration(message=AIMessage(content=text))]
    ]
    return LLMResult(generations=generations)


def test_records_a_chat_model_call() -> None:
    handler = ReenactCallbackHandler()
    run_id = uuid4()
    handler.on_chat_model_start(
        {"kwargs": {"model": "claude-sonnet-4-5"}}, _messages("hi"), run_id=run_id
    )
    handler.on_llm_end(_llm_result("Hello!"), run_id=run_id)

    event = handler.recorder.trajectory.events[0]
    assert isinstance(event, LLMCallEvent)
    assert event.provider == "langchain"
    assert event.model == "claude-sonnet-4-5"


def test_records_a_tool_call() -> None:
    handler = ReenactCallbackHandler()
    run_id = uuid4()
    handler.on_tool_start({"name": "search_docs"}, "reenact", run_id=run_id)
    handler.on_tool_end("3 hits", run_id=run_id)

    event = handler.recorder.trajectory.events[0]
    assert isinstance(event, ToolCallEvent)
    assert event.name == "search_docs"
    assert event.result == "3 hits"


def test_llm_and_tool_calls_share_the_trajectory() -> None:
    handler = ReenactCallbackHandler()
    llm_run, tool_run = uuid4(), uuid4()
    handler.on_chat_model_start({"kwargs": {}}, _messages("hi"), run_id=llm_run)
    handler.on_llm_end(_llm_result("x"), run_id=llm_run)
    handler.on_tool_start({"name": "t"}, "in", run_id=tool_run)
    handler.on_tool_end("out", run_id=tool_run)

    types = [e.type for e in handler.recorder.trajectory.events]
    assert types == ["llm_call", "tool_call"]


def test_records_a_langgraph_node_boundary_with_checkpoint_coordinates() -> None:
    handler = ReenactCallbackHandler()
    run_id = uuid4()
    # The metadata keys LangGraph actually emits per node (see test_langgraph.py).
    metadata = {
        "langgraph_node": "agent",
        "langgraph_step": "2",
        "thread_id": "t1",
        "langgraph_checkpoint_ns": "agent:abc123",
    }
    handler.on_chain_start({}, {"messages": []}, run_id=run_id, metadata=metadata)
    handler.on_chain_end({"messages": []}, run_id=run_id)

    event = handler.recorder.trajectory.events[0]
    assert isinstance(event, GraphNodeEvent)
    assert event.node == "agent"
    assert event.step == 2  # parsed from the string LangGraph reports
    assert event.thread_id == "t1"
    assert event.checkpoint_ns == "agent:abc123"


def test_ignores_ordinary_chains_without_a_langgraph_node() -> None:
    handler = ReenactCallbackHandler()
    run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=run_id, metadata={"foo": "bar"})
    handler.on_chain_end({}, run_id=run_id)

    assert handler.recorder.trajectory.events == []


def test_node_without_checkpoint_coordinates_records_none() -> None:
    handler = ReenactCallbackHandler()
    run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=run_id, metadata={"langgraph_node": "tools"})
    handler.on_chain_end({}, run_id=run_id)

    event = handler.recorder.trajectory.events[0]
    assert isinstance(event, GraphNodeEvent)
    assert event.node == "tools"
    assert event.step is None
    assert event.thread_id is None
    assert event.checkpoint_ns is None
