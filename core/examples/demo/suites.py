"""The demo eval suite as scenarios - the same checks over any recorded set.

The checks are defined once per issue, then applied to a named cassette set
(``baseline``, ``model-swap``, ...), so a test or the metrics bench can run the
identical suite against each set and diff the results. ``suite.toml`` is the
committable mirror of this for the ``reenact eval`` / ``ci`` CLI.
"""

from pathlib import Path
from typing import Any

from reenact.evals import (
    Check,
    Scenario,
    answer_contains,
    called_tool,
    judged,
    no_mutating_tool_reexecuted,
)

SCENARIOS = Path(__file__).resolve().parent / "scenarios"

# A topical keyword present in every good triage of each issue (robust across the
# clean re-record, so it never false-positives).
_ANSWER_KEYWORD = {"42": "password", "57": "billing", "63": "429"}

JUDGE_RUBRIC = (
    "The agent correctly triaged the support issue: it searched the documentation, "
    "applied exactly one appropriate category label (bug, billing, or api), posted "
    "a helpful reply grounded in the docs, and gave an accurate one-sentence "
    "summary. Score 1.0 for a complete, correct, well-grounded triage; score lower "
    "for a missing or wrong label, a skipped step, an ungrounded or unhelpful "
    "reply, or an inaccurate summary."
)


def _checks_for(issue_id: str, judge_client: Any) -> list[Check]:
    checks: list[Check] = [
        called_tool("search_docs"),
        called_tool("label_issue"),
        called_tool("post_reply"),
        no_mutating_tool_reexecuted(),
        answer_contains(_ANSWER_KEYWORD[issue_id]),
    ]
    if judge_client is not None:
        checks.append(judged(judge_client, JUDGE_RUBRIC, name="judge"))
    return checks


def demo_scenarios(set_name: str, *, judge_client: Any = None) -> list[Scenario]:
    """Build the demo scenarios for a recorded set (e.g. ``baseline``).

    The structural assertions (tool calls, mutating-tool safety, a topical keyword)
    are the gate's reliable signal. With ``judge_client`` each scenario also gets
    the trajectory judge - a supplementary quality lens, not required to reproduce
    the catch-rate / FPR numbers.
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
