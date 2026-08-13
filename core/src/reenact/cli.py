"""Reenact command-line interface, built with Typer."""

from __future__ import annotations

import importlib
import importlib.util
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import typer

from reenact import __version__
from reenact.evals import (
    Baseline,
    EvalReport,
    RegressionDiff,
    SuiteConfigError,
    diff_baselines,
    load_baseline,
    load_suite,
    run_suite,
    save_baseline,
)
from reenact.replay import Player, ReplayMode
from reenact.schema import LLMCallEvent, ToolCallEvent, Trajectory
from reenact.store import load_cassette, save_cassette

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


def _judge_client() -> Any:
    """A best-effort judge client for judge checks, or ``None`` if unavailable.

    Constructs an Anthropic client from ``ANTHROPIC_API_KEY`` when the SDK is
    installed, importing it lazily so the CLI keeps no hard dependency on it. When
    this returns ``None`` a suite with a judge check fails to load with a clear
    message; a suite of plain assertions never needs it.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        module: Any = importlib.import_module("anthropic")
    except ImportError:
        return None
    return module.Anthropic()


def _render_report(report: EvalReport, suite_path: Path) -> None:
    typer.echo(f"suite {suite_path} ({report.total} scenario(s))")
    for scenario in report.scenarios:
        passed_checks = sum(1 for check in scenario.checks if check.passed)
        status = "PASS" if scenario.passed else "FAIL"
        typer.echo(
            f"  {status} {scenario.name} "
            f"({passed_checks}/{len(scenario.checks)} checks)"
        )
        for check in scenario.failures:
            typer.echo(f"    - {check.name}: {check.message}")
    typer.echo(f"{report.passed_count}/{report.total} scenarios passed")


def _load_suite_or_exit(suite: Path) -> EvalReport:
    """Load and run a suite, exiting 2 on a config error."""
    try:
        scenarios = load_suite(suite, judge_client=_judge_client())
    except SuiteConfigError as exc:
        typer.echo(f"error: {exc}")
        raise typer.Exit(2) from exc
    return run_suite(scenarios)


@app.command("eval")
def eval_suite(
    suite: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        help="Path to a TOML eval suite.",
    ),
    write_baseline: Path | None = typer.Option(
        None,
        "--write-baseline",
        dir_okay=False,
        help="Write this run as a baseline JSON for `reenact ci` to diff against.",
    ),
) -> None:
    """Run an eval suite offline and report per-scenario pass/fail.

    Each scenario replays its recorded cassette and runs its checks - assertions
    and, where configured, a trajectory-level judge. Exits non-zero if any
    scenario fails. With ``--write-baseline`` it also records the run as the
    last-known-good snapshot the CI gate compares against.
    """
    report = _load_suite_or_exit(suite)
    _render_report(report, suite)
    if write_baseline is not None:
        save_baseline(Baseline.from_report(report), write_baseline)
        typer.echo(f"wrote baseline {write_baseline}")
    if not report.passed:
        raise typer.Exit(1)


def _render_diff(diff: RegressionDiff) -> None:
    typer.echo(diff.summary())
    for delta in diff.regressions:
        typer.echo(f"  regressed: {delta.check} {delta.detail} ({delta.scenario})")
    for delta in diff.improvements:
        typer.echo(f"  improved:  {delta.check} {delta.detail} ({delta.scenario})")
    for delta in diff.new_checks:
        typer.echo(f"  new:       {delta.check} ({delta.scenario})")


@app.command()
def ci(
    suite: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        help="Path to a TOML eval suite.",
    ),
    baseline: Path = typer.Option(
        ...,
        "--baseline",
        "-b",
        exists=True,
        dir_okay=False,
        help="Committed baseline JSON to diff against (see `eval --write-baseline`).",
    ),
    tolerance: float = typer.Option(
        0.05,
        "--tolerance",
        help="Score drop tolerated before it counts as a regression.",
    ),
) -> None:
    """Run the suite and fail only if it regressed against a committed baseline.

    Unlike ``eval`` (which fails on any check failure), ``ci`` fails on *drift*:
    a check that went pass to fail, or a judge score that dropped past the
    tolerance, relative to the baseline. Exits 1 on a regression, else 0.
    """
    report = _load_suite_or_exit(suite)
    diff = diff_baselines(
        load_baseline(baseline),
        Baseline.from_report(report),
        score_tolerance=tolerance,
    )
    _render_diff(diff)
    if diff.regressed:
        raise typer.Exit(1)


def _load_module_from_path(path: Path) -> Any:
    if not path.is_file():
        raise typer.BadParameter(f"no such file: {path}")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise typer.BadParameter(f"cannot import a module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_entrypoint(spec: str) -> Callable[[], Any]:
    """Resolve ``module:function`` or ``path.py:function`` to a callable."""
    module_part, sep, func_name = spec.rpartition(":")
    if not sep or not module_part or not func_name:
        raise typer.BadParameter(
            "entrypoint must be 'module:function' or 'path.py:function'"
        )
    if module_part.endswith(".py") or "/" in module_part or os.sep in module_part:
        module = _load_module_from_path(Path(module_part))
    else:
        module = importlib.import_module(module_part)
    func = getattr(module, func_name, None)
    if not callable(func):
        raise typer.BadParameter(f"{func_name!r} is not a callable in {module_part!r}")
    return cast(Callable[[], Any], func)


def _as_trajectory(result: Any) -> Trajectory:
    if isinstance(result, Trajectory):
        return result
    candidate = getattr(result, "trajectory", None)
    if isinstance(candidate, Trajectory):
        return candidate
    raise typer.BadParameter(
        "entrypoint must return a Trajectory or a Recorder (with a .trajectory)"
    )


@app.command()
def record(
    entrypoint: str = typer.Argument(
        ...,
        help="'module:function' or 'path.py:function' returning a Trajectory.",
    ),
    output: Path = typer.Argument(
        ...,
        dir_okay=False,
        help="Where to write the cassette JSON.",
    ),
) -> None:
    """Run a scenario entrypoint and write its trajectory as a cassette.

    The entrypoint is a zero-argument callable that returns a Trajectory (or a
    Recorder). The capture itself happens inside it via ``reenact.recording``;
    this verb just resolves it, runs it, and writes the committable cassette.
    """
    func = _resolve_entrypoint(entrypoint)
    trajectory = _as_trajectory(func())
    output.parent.mkdir(parents=True, exist_ok=True)
    save_cassette(trajectory, output)
    typer.echo(f"wrote {output} ({len(trajectory.events)} event(s))")


if __name__ == "__main__":
    app()
