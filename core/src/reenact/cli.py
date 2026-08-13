"""Reenact command-line interface, built with Typer."""

from __future__ import annotations

from pathlib import Path

import typer

from reenact import __version__
from reenact.replay import Player, ReplayMode
from reenact.schema import LLMCallEvent, ToolCallEvent
from reenact.store import load_cassette

app = typer.Typer(
    name="reenact",
    help="Regression testing for LLM agents.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"reenact {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the Reenact version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Reenact — regression testing for LLM agents."""


@app.command()
def replay(
    cassette: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        help="Path to a recorded cassette (JSON).",
    ),
) -> None:
    """Replay a recorded run offline and report whether it reproduces byte-identical.

    Feeds every captured call back through the substitution engine with no
    network: each recorded request must reproduce its recorded response, and the
    cassette must re-serialize byte-for-byte. Exits non-zero if the run diverges.
    """
    trajectory = load_cassette(cassette)
    player = Player(trajectory, mode=ReplayMode.LENIENT)
    counts: dict[str, int] = {}
    for event in trajectory.events:
        counts[event.type] = counts.get(event.type, 0) + 1
        if isinstance(event, LLMCallEvent):
            player.replay_llm_call(event.request)
        elif isinstance(event, ToolCallEvent):
            player.replay_tool_call(event.name, event.arguments)

    reserialized = trajectory.model_dump_json(indent=2) + "\n"
    round_trip = reserialized == cassette.read_text(encoding="utf-8")
    summary = ", ".join(f"{n} {t}" for t, n in sorted(counts.items())) or "no events"

    typer.echo(f"replaying {trajectory.name or trajectory.id} ({summary})")
    typer.echo(f"  offline replay: {len(player.divergences)} divergence(s)")
    typer.echo(f"  round-trip byte-identical: {'yes' if round_trip else 'no'}")
    if player.divergences or not round_trip:
        for divergence in player.divergences:
            typer.echo(f"  - {divergence.message}")
        raise typer.Exit(1)
    typer.echo("clean - reproduced offline with no network, $0")


if __name__ == "__main__":
    app()
