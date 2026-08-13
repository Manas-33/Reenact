"""Record the weather agent against the live Anthropic API into a golden cassette.

Run this once, with a real key in the environment, to (re)generate
``examples/golden/weather.json``. That committed cassette is what the offline
tests replay - regenerating it needs network and a key, but the tests never do.

    export ANTHROPIC_API_KEY=sk-ant-...      # see .env.example at the repo root
    python examples/record_weather.py

The recorder redacts secrets before writing, so the cassette is safe to commit.
"""

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from weather_agent import run_agent

import reenact
from reenact.schema import SideEffect, Trajectory
from reenact.store import save_cassette

GOLDEN = Path(__file__).resolve().parent / "golden" / "weather.json"

# Pinned so regenerating the fixture diffs only when the recorded behavior
# changes - not on every run from a fresh uuid and timestamp.
_FIXTURE_ID = "example-weather"
_FIXTURE_TIME = datetime(2024, 1, 1, tzinfo=UTC)


def record(client: Any) -> Trajectory:
    """Run the agent through ``client``, recording its LLM and tool calls."""
    with reenact.recording(client) as rec:

        def _on_tool(name: str, arguments: dict[str, Any], result: str) -> None:
            rec.record_tool_call(
                name=name,
                arguments=arguments,
                result=result,
                side_effect=SideEffect.READ_ONLY,
            )

        run_agent(client, on_tool=_on_tool)

    trajectory = rec.trajectory
    trajectory.name = "weather"
    trajectory.id = _FIXTURE_ID
    trajectory.created_at = _FIXTURE_TIME
    return trajectory


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env, add your "
            "key, and export it before recording."
        )
    import anthropic

    trajectory = record(anthropic.Anthropic())
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    save_cassette(trajectory, GOLDEN)
    print(f"wrote {GOLDEN}")


if __name__ == "__main__":
    main()
