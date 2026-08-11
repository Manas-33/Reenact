"""Smoke test: the package imports, exposes a version, and the CLI runs."""

from __future__ import annotations

from typer.testing import CliRunner

import reenact
from reenact.cli import app


def test_package_exposes_version() -> None:
    assert reenact.__version__ == "0.0.1"


def test_cli_version_flag_runs() -> None:
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.0.1" in result.stdout
