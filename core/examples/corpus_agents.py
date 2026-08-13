"""A small corpus of real agent shapes for the determinism benchmark.

Each agent knows nothing about reenact: it takes any object exposing
``messages.create(**kwargs)`` - a live ``anthropic.Anthropic`` client when
recording, ``reenact.replaying`` when replaying offline - plus, where noted, an
``on_tool`` hook the recorder uses to log tool calls and a reenact ``Clock`` so a
timestamped agent replays byte-identical.

The shapes deliberately differ, so the corpus is "not one agent repeated": a
single call (``qa``), a tool round-trip (``tool_use``), a parallel tool fan-out
(``parallel`` - several tool_use blocks in one turn, a replay window), a
multi-step loop (``multistep``), and a clock-stamped prompt (``timestamped``).
"""

from collections.abc import Callable
from typing import Any

from reenact.replay import Clock

MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 512

# Called with (tool_name, arguments, result) after each tool invocation.
ToolHook = Callable[[str, dict[str, Any], str], None]

_FORECASTS = {
    "Paris": "18C and cloudy",
    "London": "12C and rainy",
    "Tokyo": "24C and clear",
    "Cairo": "33C and sunny",
    "Oslo": "5C and snowy",
    "Lima": "20C and foggy",
    "Nairobi": "26C and humid",
    "Reykjavik": "3C and windy",
}

WEATHER_TOOL: dict[str, Any] = {
    "name": "get_weather",
    "description": "Look up the current weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


def get_weather(city: str) -> str:
    """A pure, deterministic stand-in tool - safe to re-run on replay."""
    return _FORECASTS.get(city, "unknown")


def _text(body: dict[str, Any]) -> str:
    content: Any = body.get("content", [])
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _tool_uses(body: dict[str, Any]) -> list[dict[str, Any]]:
    content: Any = body.get("content", [])
    return [
        block
        for block in content
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]


def _create(client: Any, **request: Any) -> dict[str, Any]:
    body: dict[str, Any] = client.messages.create(**request).model_dump(mode="json")
    return body


def _tool_round(client: Any, question: str, on_tool: ToolHook | None) -> str:
    """One think -> act(s) -> think round; handles one or many tool_use blocks."""
    first = _create(
        client,
        model=MODEL,
        max_tokens=MAX_TOKENS,
        tools=[WEATHER_TOOL],
        messages=[{"role": "user", "content": question}],
    )
    uses = _tool_uses(first)
    if not uses:
        return _text(first)
    results: list[dict[str, Any]] = []
    for use in uses:
        args: dict[str, Any] = use.get("input", {})
        result = get_weather(str(args.get("city", "")))
        if on_tool is not None:
            on_tool(str(use.get("name", "")), args, result)
        results.append(
            {"type": "tool_result", "tool_use_id": use.get("id"), "content": result}
        )
    second = _create(
        client,
        model=MODEL,
        max_tokens=MAX_TOKENS,
        tools=[WEATHER_TOOL],
        messages=[
            {"role": "user", "content": question},
            {"role": "assistant", "content": first["content"]},
            {"role": "user", "content": results},
        ],
    )
    return _text(second)


def qa(client: Any, question: str) -> str:
    """One model call, no tools."""
    return _text(
        _create(
            client,
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": question}],
        )
    )


def tool_use(client: Any, city: str, on_tool: ToolHook | None = None) -> str:
    """think -> act -> think around a single tool call."""
    question = f"What's the weather in {city}? Use the get_weather tool."
    return _tool_round(client, question, on_tool)


def parallel(client: Any, cities: list[str], on_tool: ToolHook | None = None) -> str:
    """One model turn fans out into a tool call per city - a parallel window."""
    listing = ", ".join(cities)
    question = (
        f"Report the current weather for each of these cities: {listing}. "
        "Call get_weather once for every city."
    )
    return _tool_round(client, question, on_tool)


def multistep(client: Any, topic: str) -> str:
    """A three-call reasoning loop; each call consumes the previous answer."""
    aspects = _text(
        _create(
            client,
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": f"List two key aspects of {topic}."}],
        )
    )
    connection = _text(
        _create(
            client,
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[
                {
                    "role": "user",
                    "content": f"Given these aspects:\n{aspects}\n"
                    "Write one sentence connecting them.",
                }
            ],
        )
    )
    summary = f"Summarize in one line: {connection}"
    return _text(
        _create(
            client,
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": summary}],
        )
    )


def timestamped(client: Any, clock: Clock, question: str) -> str:
    """Stamps the current time (read through a reenact Clock) into the prompt.

    Reading the clock through the shim is what lets the recording replay
    byte-identical: the live run logs the real time, replay feeds it back.
    """
    system = f"The current time is {clock.now().isoformat()}."
    return _text(
        _create(
            client,
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": question}],
        )
    )
