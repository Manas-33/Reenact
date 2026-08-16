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

_ROOT = Path(__file__).resolve().parents[2]
ACTION = _ROOT / "action" / "action.yml"
WORKFLOW = _ROOT / ".github" / "workflows" / "reenact-demo.yml"


def _load() -> dict[str, Any]:
    data = yaml.safe_load(ACTION.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return cast(dict[str, Any], data)


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
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


# --- marketplace branding + the post step ------------------------------------


def test_action_has_marketplace_branding() -> None:
    branding = _load()["branding"]
    assert branding["icon"]
    assert branding["color"]


def test_token_input_defaults_to_empty() -> None:
    assert _load()["inputs"]["token"]["default"] == ""


def test_ci_step_emits_json_for_the_post_step() -> None:
    ci = next(step["run"] for step in _steps() if "reenact ci" in step.get("run", ""))
    assert "--json" in ci
    assert "reenact-diff.json" in ci


def test_post_step_reports_always_and_only_with_a_token() -> None:
    post = next(s for s in _steps() if "reenact report" in s.get("run", ""))
    # Runs even when the gate failed (to still comment), and only when a token is set.
    assert "always()" in post["if"]
    assert "inputs.token" in post["if"]
    assert post["env"]["GITHUB_TOKEN"] == "${{ inputs.token }}"
    # Reads the very diff the ci step wrote.
    assert "reenact-diff.json" in post["run"]


# --- the demo gate workflow --------------------------------------------------


def test_demo_workflow_gates_prs_with_the_local_action() -> None:
    workflow = _load_yaml(WORKFLOW)
    # PyYAML (YAML 1.1) parses the bare key `on:` as the boolean True, not "on".
    keyed = cast(dict[object, Any], workflow)
    triggers = keyed.get(True, keyed.get("on"))
    assert triggers is not None
    if isinstance(triggers, str):
        assert triggers == "pull_request"
    else:
        assert "pull_request" in triggers

    permissions = workflow["permissions"]
    assert permissions["pull-requests"] == "write"
    assert permissions["checks"] == "write"

    steps = workflow["jobs"]["gate"]["steps"]
    gate = next(step for step in steps if step.get("uses") == "./action")
    assert gate["with"]["version"] == "./core"  # installs from the checkout, no PyPI
    assert "demo/suite.toml" in gate["with"]["suite"]
    assert gate["with"]["token"] == "${{ github.token }}"
