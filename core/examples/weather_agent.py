"""A tiny tool-using agent for the record/replay end-to-end example.

It knows nothing about reenact. It takes any object exposing
``messages.create(**kwargs)`` - a live ``anthropic.Anthropic`` client when
recording, ``reenact.replaying`` when replaying offline - plus an optional
``on_tool`` hook that reenact uses to record the tool call. The single tool is a
pure, deterministic function, so replay can safely re-run it.

The run is two model calls around one tool call: a think -> act -> think
trajectory, so the recording exercises the interleaved event ordering, not just
a single model call.
"""

from collections.abc import Callable
from typing import Any

MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 256
QUESTION = "What's the weather in Paris? Use the get_weather tool."

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_weather",
        "description": "Look up the current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]

_FORECASTS = {"Paris": "18C and cloudy", "London": "12C and rainy"}

# Called with (tool_name, arguments, result) after each tool invocation.
ToolHook = Callable[[str, dict[str, Any], str], None]


def get_weather(city: str) -> str:
    """A pure stand-in tool: no network, deterministic, safe to re-run on replay."""
    return _FORECASTS.get(city, "unknown")


def _first_tool_use(body: dict[str, Any]) -> dict[str, Any] | None:
    content: Any = body.get("content", [])
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            return block
    return None


def _text(body: dict[str, Any]) -> str:
    content: Any = body.get("content", [])
    parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "".join(parts)


def run_agent(client: Any, on_tool: ToolHook | None = None) -> str:
    """Ask about the weather, let the model call the tool, return the final text."""
    first: dict[str, Any] = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        tools=TOOLS,
        messages=[{"role": "user", "content": QUESTION}],
    ).model_dump(mode="json")

    tool_use = _first_tool_use(first)
    if tool_use is None:
        return _text(first)

    arguments: dict[str, Any] = tool_use.get("input", {})
    result = get_weather(str(arguments.get("city", "")))
    if on_tool is not None:
        on_tool(str(tool_use.get("name", "")), arguments, result)

    second: dict[str, Any] = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        tools=TOOLS,
        messages=[
            {"role": "user", "content": QUESTION},
            {"role": "assistant", "content": first["content"]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.get("id"),
                        "content": result,
                    }
                ],
            },
        ],
    ).model_dump(mode="json")
    return _text(second)
