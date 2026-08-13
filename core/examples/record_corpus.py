"""Record the agent corpus against the live Anthropic API into committed goldens.

Run once, with a real key, to (re)generate ``examples/corpus/*.json`` - the real
recordings the determinism bench and the corpus tests replay offline. Every run
here spends tokens; the tests never do.

    export ANTHROPIC_API_KEY=sk-ant-...        # see .env.example at the repo root
    python examples/record_corpus.py            # full corpus (>=100 runs)
    python examples/record_corpus.py --smoke    # one run per shape (a cheap check)

Ids and timestamps are pinned, so regenerating diffs only where the recorded
model text changed - not on every fresh uuid/clock.
"""

import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import corpus_agents as agents

import reenact
from reenact.record import Recorder
from reenact.replay import Clock, save_entropy
from reenact.schema import SideEffect, Trajectory
from reenact.store import save_cassette

CORPUS = Path(__file__).resolve().parent / "corpus"
_FIXTURE_TIME = datetime(2024, 1, 1, tzinfo=UTC)

# Scenario inputs per shape. Sliced to one each in --smoke mode.
QUESTIONS = [
    "What is the capital of France?",
    "Name a primary color.",
    "What is 12 times 12?",
    "Who wrote Hamlet?",
    "What gas do plants absorb?",
    "What is the largest planet?",
    "How many continents are there?",
    "What is the freezing point of water in Celsius?",
    "Name the closest star to Earth.",
    "What language is spoken in Brazil?",
    "What is the square root of 81?",
    "Which ocean is the largest?",
    "What metal is liquid at room temperature?",
    "How many sides does a hexagon have?",
    "What is the chemical symbol for gold?",
    "Who painted the Mona Lisa?",
    "What is the tallest mountain on Earth?",
    "What year did the first human land on the Moon?",
    "What is the smallest prime number?",
    "What organ pumps blood through the body?",
]
CITIES = [
    "Paris", "London", "Tokyo", "Cairo", "Oslo", "Lima", "Nairobi", "Reykjavik",
    "Paris", "London", "Tokyo", "Cairo", "Oslo", "Lima", "Nairobi", "Reykjavik",
    "Paris", "London", "Tokyo", "Cairo",
]
CITY_SETS = [
    ["Paris", "London", "Tokyo"],
    ["Cairo", "Oslo"],
    ["Lima", "Nairobi", "Reykjavik"],
    ["Paris", "Cairo", "Oslo"],
    ["London", "Tokyo"],
    ["Nairobi", "Lima"],
    ["Reykjavik", "Paris", "London"],
    ["Tokyo", "Cairo", "Nairobi"],
    ["Oslo", "Lima"],
    ["Paris", "Reykjavik"],
    ["London", "Cairo", "Tokyo"],
    ["Nairobi", "Oslo", "Paris"],
    ["Lima", "Tokyo"],
    ["Cairo", "Reykjavik", "London"],
    ["Paris", "Nairobi"],
    ["Oslo", "Tokyo", "Cairo"],
    ["London", "Lima", "Reykjavik"],
    ["Tokyo", "Paris"],
    ["Nairobi", "Cairo"],
    ["Oslo", "London", "Lima"],
]
TOPICS = [
    "photosynthesis", "supply and demand", "the water cycle", "machine learning",
    "plate tectonics", "compound interest", "natural selection", "the internet",
    "climate change", "democracy", "gravity", "vaccination", "inflation",
    "black holes", "the immune system", "renewable energy", "encryption",
    "the stock market", "antibiotics", "electric motors",
]
TS_QUESTIONS = [
    "What day of the week is it, roughly?",
    "Is it morning or evening right now?",
    "Give a one-line greeting for the current time.",
    "What season is it likely to be?",
    "Suggest a meal for this time of day.",
    "Is it a workday or weekend, probably?",
    "What is a good activity for right now?",
    "Say hello referencing the hour.",
    "Is it late or early?",
    "Recommend a drink for this time.",
    "What is a fitting one-word mood for now?",
    "Should someone be asleep at this hour?",
    "Name a task suited to this time.",
    "Give a time-appropriate farewell.",
    "Is the sun likely up right now?",
    "What is a good reminder for this hour?",
    "Suggest music for this time of day.",
    "Is it close to lunchtime?",
    "Offer a short motivational line for now.",
    "What is a sensible next hour to plan?",
]


def _finalize(trajectory: Trajectory, shape: str, index: int) -> Path:
    trajectory.name = shape
    trajectory.id = f"{shape}-{index:02d}"
    trajectory.created_at = _FIXTURE_TIME
    path = CORPUS / f"{shape}-{index:02d}.json"
    save_cassette(trajectory, path)
    return path


def _on_tool(rec: Recorder) -> agents.ToolHook:
    def hook(name: str, arguments: dict[str, Any], result: str) -> None:
        rec.record_tool_call(
            name=name,
            arguments=arguments,
            result=result,
            side_effect=SideEffect.READ_ONLY,
        )

    return hook


def record_qa(client: Any, index: int, question: str) -> Path:
    with reenact.recording(client) as rec:
        agents.qa(client, question)
    return _finalize(rec.trajectory, "qa", index)


def record_tool_use(client: Any, index: int, city: str) -> Path:
    with reenact.recording(client) as rec:
        agents.tool_use(client, city, on_tool=_on_tool(rec))
    return _finalize(rec.trajectory, "tool_use", index)


def record_parallel(client: Any, index: int, cities: list[str]) -> Path:
    with reenact.recording(client) as rec:
        agents.parallel(client, cities, on_tool=_on_tool(rec))
    return _finalize(rec.trajectory, "parallel", index)


def record_multistep(client: Any, index: int, topic: str) -> Path:
    with reenact.recording(client) as rec:
        agents.multistep(client, topic)
    return _finalize(rec.trajectory, "multistep", index)


def record_timestamped(client: Any, index: int, question: str) -> Path:
    clock = Clock()
    with reenact.recording(client) as rec:
        agents.timestamped(client, clock, question)
    save_entropy(rec.trajectory, clock=clock)
    return _finalize(rec.trajectory, "timestamped", index)


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set; export it before recording.")
    import anthropic

    smoke = "--smoke" in sys.argv
    limit = 1 if smoke else None
    client = anthropic.Anthropic()
    CORPUS.mkdir(parents=True, exist_ok=True)

    plan: list[tuple[str, Any, list[Any]]] = [
        ("qa", record_qa, QUESTIONS[:limit]),
        ("tool_use", record_tool_use, CITIES[:limit]),
        ("parallel", record_parallel, CITY_SETS[:limit]),
        ("multistep", record_multistep, TOPICS[:limit]),
        ("timestamped", record_timestamped, TS_QUESTIONS[:limit]),
    ]

    total = 0
    failures = 0
    for shape, record, scenarios in plan:
        for index, scenario in enumerate(scenarios):
            try:
                path = record(client, index, scenario)
                total += 1
                print(f"  {path.name}")
            except Exception as exc:  # report the failure and keep going
                failures += 1
                print(f"  FAILED {shape}-{index:02d}: {exc}")
    print(f"wrote {total} recordings to {CORPUS} ({failures} failed)")


if __name__ == "__main__":
    main()
