"""The demo eval suite as scenarios - the same checks over any recorded set.

The checks are defined once per issue, then applied to a named cassette set
(``baseline``, ``model-swap``, ...), so a test or the metrics bench can run the
identical suite against each set and diff the results. ``suite.toml`` is the
committable mirror of this for the ``reenact eval`` / ``ci`` CLI.
"""

from pathlib import Path
from typing import Any

from reenact.evals import (
    FAITHFULNESS,
    Check,
    Criterion,
    Scenario,
    answer_contains,
    called_tool,
    no_mutating_tool_reexecuted,
    structured_eval,
)

SCENARIOS = Path(__file__).resolve().parent / "scenarios"

# A topical keyword present in every good triage of each issue (robust across the
# clean re-record, so it never false-positives).
_ANSWER_KEYWORD = {"42": "password", "57": "billing", "63": "429"}

# The fuzzy-quality half, replacing the scalar judge: evidence-backed criteria,
# each a yes/no question the model answers with a citation to a trajectory step,
# gated per-criterion exactly like a hard assertion (no float, no threshold to
# tune). Authored per the agent's job, not as a universal constant.
QUALITY_CRITERIA = [
    Criterion(
        id="correct_label",
        question=(
            "Did the agent apply exactly one appropriate category label "
            "(bug, billing, or api) matching this issue?"
        ),
    ),
    Criterion(
        id="reply_grounded",
        question=(
            "Is the reply the agent posted grounded in the documentation it "
            "retrieved during the run, rather than invented?"
        ),
    ),
    FAITHFULNESS,
]


def _checks_for(issue_id: str, judge_client: Any) -> list[Check]:
    checks: list[Check] = [
        called_tool("search_docs"),
        called_tool("label_issue"),
        called_tool("post_reply"),
        no_mutating_tool_reexecuted(),
        answer_contains(_ANSWER_KEYWORD[issue_id]),
    ]
    if judge_client is not None:
        checks.extend(structured_eval(judge_client, QUALITY_CRITERIA))
    return checks


def demo_scenarios(set_name: str, *, judge_client: Any = None) -> list[Scenario]:
    """Build the demo scenarios for a recorded set (e.g. ``baseline``).

    The structural assertions (tool calls, mutating-tool safety, a topical keyword)
    are the gate's reliable signal. With ``judge_client`` each scenario also gets
    the structured criteria - evidence-backed soft assertions, a supplementary
    quality lens that is not required to reproduce the catch-rate / FPR numbers.
    """
    base = SCENARIOS / set_name
    return [
        Scenario.from_cassette(
            base / f"issue-{issue_id}.json",
            _checks_for(issue_id, judge_client),
            name=f"triage-{issue_id}",
        )
        for issue_id in _ANSWER_KEYWORD
    ]
