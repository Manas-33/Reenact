"""Reenact command-line interface, built with Typer."""

from __future__ import annotations

import typer

from reenact import __version__

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


if __name__ == "__main__":
    app()
