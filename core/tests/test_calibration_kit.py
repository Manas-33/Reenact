"""The calibration labelling kit: the borderline probes and the sheet generator.

Keyless: builds the sheets over the committed demo runs plus the synthetic stress
probes and checks the grid is the right size, the human sheet is blind (no verdicts
leak in), the evaluator prompt asks for the JSON shape, and a probe's deliberately-
borderline detail actually renders so the hard case is really on the sheet.
"""

import json
from pathlib import Path

from demo.calibration.probes import PROBE_TARGETS, probes
from demo.calibration.sheet import (
    build_items,
    collect_trajectories,
    render_evaluator_prompt,
    render_human_sheet,
    write_sheets,
)
from demo.suites import QUALITY_CRITERIA

from reenact.schema import LLMCallEvent, ToolCallEvent


def test_probes_are_triage_runs_with_the_target_detail() -> None:
    by_name = {p.name: p for p in probes()}
    assert set(by_name) == set(PROBE_TARGETS)
    # The ungrounded-reply probe must actually contain the unsupported claim, or it
    # is not borderline. Its reply mentions the mobile app; the doc never does.
    replies = [
        e.arguments["body"]
        for e in by_name["probe-ungrounded-reply"].events
        if isinstance(e, ToolCallEvent) and e.name == "post_reply"
    ]
    assert any("mobile app" in reply for reply in replies)
    # The overstated-summary probe claims an action (raising the limit) no tool did.
    final = by_name["probe-overstated-summary"].events[-1]
    assert isinstance(final, LLMCallEvent)
    assert "raised your account's rate limit" in final.response["content"][0]["text"]


def test_grid_is_real_runs_plus_stress_probes() -> None:
    trajectories = collect_trajectories()
    groups = [st.group for st in trajectories]
    assert groups.count("real") == 15  # 5 sets x 3 issues
    assert groups.count("stress") == 3  # the probes
    items = build_items(trajectories)
    assert len(items) == 18 * len(QUALITY_CRITERIA)
    assert len({i["id"] for i in items}) == len(items)  # ids are unique
    assert {i["group"] for i in items} == {"real", "stress"}


def test_human_sheet_is_blind() -> None:
    sheet = render_human_sheet(collect_trajectories())
    # Every criterion question is present, with an empty fill slot per item...
    for criterion in QUALITY_CRITERIA:
        assert criterion.question in sheet
        assert f"`baseline/issue-42::{criterion.id}` = " in sheet
    # ...and nothing reveals a verdict or which runs are the stress cases.
    assert "passed" not in sheet
    assert "stress" not in sheet
    assert "group" not in sheet


def test_evaluator_prompt_requests_json_with_evidence() -> None:
    prompt = render_evaluator_prompt(collect_trajectories())
    assert '"passed"' in prompt and '"evidence"' in prompt
    assert "JSON array" in prompt
    # Item ids are listed for the model to key its answers on.
    assert "probe/probe-debatable-label::correct_label" in prompt


def test_write_sheets_emits_all_three_files(tmp_path: Path) -> None:
    paths = write_sheets(tmp_path)
    assert paths["human"].is_file()
    assert paths["evaluator"].is_file()
    manifest = json.loads(paths["items"].read_text(encoding="utf-8"))
    assert len(manifest["items"]) == 18 * len(QUALITY_CRITERIA)
    assert [c["id"] for c in manifest["criteria"]] == [c.id for c in QUALITY_CRITERIA]
