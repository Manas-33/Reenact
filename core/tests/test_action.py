"""The composite GitHub Action that wraps the reenact CLI.

A GitHub Action can't run on a local runner, so this validates ``action.yml``'s
contract instead: it is a well-formed composite action, its inputs wire through to
``reenact ci``, and every composite ``run`` step declares a ``shell`` (a required
field GitHub otherwise rejects only at run time). This is the offline half of the
gate; exercising it against a real PR is a launch step.
"""

from pathlib import Path
from typing import Any, cast

import yaml

ACTION = Path(__file__).resolve().parents[2] / "action" / "action.yml"


def _load() -> dict[str, Any]:
    data = yaml.safe_load(ACTION.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return cast(dict[str, Any], data)


def _steps() -> list[dict[str, Any]]:
    runs = cast(dict[str, Any], _load()["runs"])
    steps = runs["steps"]
    assert isinstance(steps, list)
    return cast(list[dict[str, Any]], steps)


def test_action_is_a_composite_with_name_and_description() -> None:
    action = _load()
    assert isinstance(action.get("name"), str) and action["name"]
    assert isinstance(action.get("description"), str) and action["description"]
    assert action["runs"]["using"] == "composite"


def test_required_and_defaulted_inputs() -> None:
    inputs = _load()["inputs"]
    assert inputs["suite"]["required"] is True
    assert inputs["baseline"]["required"] is True
    assert inputs["tolerance"]["default"] == "0.05"
    # Optional knobs carry sensible defaults so a caller can set only suite+baseline.
    assert inputs["working-directory"]["default"] == "."
    assert inputs["version"]["default"] == "reenact"


def test_ci_step_wires_the_inputs() -> None:
    runs = [step["run"] for step in _steps() if "run" in step]
    ci = next(run for run in runs if "reenact ci" in run)
    assert "inputs.suite" in ci
    assert "--baseline" in ci and "inputs.baseline" in ci
    assert "--tolerance" in ci and "inputs.tolerance" in ci


def test_every_composite_run_step_declares_a_shell() -> None:
    for step in _steps():
        if "run" in step:
            assert step.get("shell"), f"composite run step needs a shell: {step}"


def test_setup_and_install_use_their_inputs() -> None:
    rendered = yaml.safe_dump(_steps())
    assert "actions/setup-python" in rendered
    assert "inputs.python-version" in rendered
    assert "inputs.version" in rendered  # the install step installs the chosen spec
