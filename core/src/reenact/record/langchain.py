"""Record LangChain and LangGraph runs via a callback handler.

Requires ``langchain-core`` (``pip install reenact[langchain]``). Pass the
handler to any LangChain or LangGraph run via ``config={"callbacks": [handler]}``;
captured LLM and tool calls land in ``handler.recorder.trajectory``. LangGraph
is covered because it runs on the same LangChain callback system.
"""

from typing import Any, cast
from uuid import UUID

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "reenact's LangChain integration requires langchain-core. "
        "Install it with: pip install reenact[langchain]"
    ) from exc

from reenact.record.recorder import Recorder
from reenact.schema import SideEffect


def _model_name(serialized: dict[str, Any]) -> str:
    kwargs = serialized.get("kwargs")
    if not isinstance(kwargs, dict):
        return ""
    typed = cast(dict[str, Any], kwargs)
    for key in ("model", "model_name", "model_id"):
        value = typed.get(key)
        if isinstance(value, str):
            return value
    return ""


def _tool_name(serialized: dict[str, Any]) -> str:
    name: Any = serialized.get("name")
    return name if isinstance(name, str) else ""


def _graph_node(metadata: dict[str, Any] | None) -> str | None:
    """The LangGraph node name for this chain, or None for an ordinary chain.

    LangGraph tags each node's run with ``langgraph_node`` in the callback
    metadata; its presence is what distinguishes a graph node boundary from the
    many ordinary Runnable chains that also fire ``on_chain_start``.
    """
    if not isinstance(metadata, dict):
        return None
    node: Any = metadata.get("langgraph_node")
    return node if isinstance(node, str) else None


def _str_field(metadata: dict[str, Any], key: str) -> str | None:
    value: Any = metadata.get(key)
    return value if isinstance(value, str) else None


def _step(metadata: dict[str, Any]) -> int | None:
    """The superstep number, which LangGraph reports as a string."""
    value: Any = metadata.get("langgraph_step")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


class ReenactCallbackHandler(BaseCallbackHandler):
    """A LangChain callback handler that records LLM and tool calls.

    Captured calls accumulate in ``self.recorder.trajectory``. Works for
    LangGraph too, since LangGraph runs on LangChain's callbacks.
    """

    def __init__(self, recorder: Recorder | None = None) -> None:
        super().__init__()
        self.recorder = recorder if recorder is not None else Recorder()
        self._pending: dict[UUID, dict[str, Any]] = {}
        self._pending_nodes: dict[UUID, dict[str, Any]] = {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        first: list[BaseMessage] = messages[0] if messages else []
        self._pending[run_id] = {
            "model": _model_name(serialized),
            "messages": [message.model_dump() for message in first],
        }

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        request: dict[str, Any] = self._pending.pop(run_id, None) or {"messages": []}
        self.recorder.record_llm_call(
            provider="langchain",
            model=str(request.get("model", "")),
            request=request,
            response=response.model_dump(),
        )

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        node = _graph_node(metadata)
        if node is None or metadata is None:
            return  # an ordinary Runnable chain, not a LangGraph node boundary
        self._pending_nodes[run_id] = {
            "node": node,
            "step": _step(metadata),
            "thread_id": _str_field(metadata, "thread_id"),
            "checkpoint_ns": _str_field(metadata, "langgraph_checkpoint_ns"),
        }

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        pending = self._pending_nodes.pop(run_id, None)
        if pending is None:
            return  # not a node boundary we opened in on_chain_start
        self.recorder.record_graph_node(
            node=str(pending["node"]),
            step=pending["step"],
            thread_id=pending["thread_id"],
            checkpoint_ns=pending["checkpoint_ns"],
        )

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._pending[run_id] = {"name": _tool_name(serialized), "input": input_str}

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        pending: dict[str, Any] = self._pending.pop(run_id, None) or {}
        result: Any = output.model_dump() if hasattr(output, "model_dump") else output
        self.recorder.record_tool_call(
            name=str(pending.get("name", "")),
            arguments={"input": pending.get("input")},
            result=result,
            side_effect=SideEffect.UNKNOWN,
        )
