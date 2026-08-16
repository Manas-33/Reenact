"""`reenact init`: scaffolding the on-ramp files, safely.

Keyless and offline - the templates are plain strings. These check that the files
are written (and skipped, not clobbered, on a re-run), that each template is
well-formed (record.py compiles, suite.toml is valid TOML, the workflow parses and
wires the action), and that the CLI reports what it did plus the next steps.
"""

import tomllib
from pathlib import Path
from typing import Any, cast

import yaml
from typer.testing import CliRunner

from reenact.cli import app
from reenact.scaffold import scaffold

runner = CliRunner()

RELATIVE_FILES = (
    "evals/record.py",
    "evals/suite.toml",
    ".github/workflows/reenact.yml",
)


# --- scaffold behavior -------------------------------------------------------


def test_scaffold_writes_all_files(tmp_path: Path) -> None:
    results = scaffold(tmp_path)
    assert all(result.written for result in results)
    for relative in RELATIVE_FILES:
        assert (tmp_path / relative).is_file(), relative


def test_scaffold_skips_existing_without_force(tmp_path: Path) -> None:
    existing = tmp_path / "evals" / "record.py"
    existing.parent.mkdir(parents=True)
    existing.write_text("my own file", encoding="utf-8")

    results = scaffold(tmp_path)
    written = {result.path.name: result.written for result in results}
    assert written["record.py"] is False  # left alone
    assert existing.read_text(encoding="utf-8") == "my own file"  # not clobbered
    # The others are still created.
    assert (tmp_path / "evals" / "suite.toml").is_file()
    assert (tmp_path / ".github" / "workflows" / "reenact.yml").is_file()


def test_scaffold_force_overwrites(tmp_path: Path) -> None:
    existing = tmp_path / "evals" / "record.py"
    existing.parent.mkdir(parents=True)
    existing.write_text("my own file", encoding="utf-8")

    scaffold(tmp_path, force=True)
    assert "reenact.recording(" in existing.read_text(encoding="utf-8")


# --- template well-formedness ------------------------------------------------


def test_record_template_compiles_and_has_the_wrapper(tmp_path: Path) -> None:
    scaffold(tmp_path)
    text = (tmp_path / "evals" / "record.py").read_text(encoding="utf-8")
    # It is valid Python (the TODOs are ellipses/comments, not syntax errors).
    compile(text, "record.py", "exec")
    assert "reenact.recording(" in text
    assert "save_cassette(" in text
    assert "TODO" in text


def test_suite_template_is_valid_toml(tmp_path: Path) -> None:
    scaffold(tmp_path)
    text = (tmp_path / "evals" / "suite.toml").read_text(encoding="utf-8")
    assert isinstance(tomllib.loads(text), dict)  # parses (a commented skeleton)
    assert "reenact suggest" in text  # points the user at the deriver


def test_workflow_parses_and_wires_the_action(tmp_path: Path) -> None:
    scaffold(tmp_path)
    workflow = tmp_path / ".github" / "workflows" / "reenact.yml"
    data = cast(dict[str, Any], yaml.safe_load(workflow.read_text(encoding="utf-8")))

    # PyYAML (YAML 1.1) parses the bare key `on:` as the boolean True, not "on".
    keyed = cast(dict[object, Any], data)
    assert keyed.get(True, keyed.get("on")) == "pull_request"
    permissions = cast(dict[str, Any], data["permissions"])
    assert permissions["pull-requests"] == "write"
    assert permissions["checks"] == "write"

    jobs = cast(dict[str, Any], data["jobs"])
    steps = cast(list[dict[str, Any]], cast(dict[str, Any], jobs["gate"])["steps"])
    action_step = next(step for step in steps if "with" in step)
    uses = cast(str, action_step["uses"])
    assert "reenact" in uses and "action" in uses
    inputs = cast(dict[str, Any], action_step["with"])
    assert inputs["suite"] == "evals/suite.toml"
    assert inputs["baseline"] == "evals/baseline.json"


# --- CLI ---------------------------------------------------------------------


def test_init_cli_creates_files_and_prints_next_steps(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0, result.stdout
    assert "wrote" in result.stdout
    assert "reenact suggest" in result.stdout  # the next-steps flow is printed
    assert (tmp_path / "evals" / "suite.toml").is_file()


def test_init_cli_skips_existing_on_a_second_run(tmp_path: Path) -> None:
    runner.invoke(app, ["init", str(tmp_path)])
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0, result.stdout
    assert "skipped (exists)" in result.stdout
