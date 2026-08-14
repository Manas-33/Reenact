"""The demo eval suite as scenarios - the same checks over any recorded set.

The checks are defined once per issue, then applied to a named cassette set
(``baseline``, ``model-swap``, ...), so a test or the metrics bench can run the
identical suite against each set and diff the results. ``suite.toml`` is the
committable mirror of this for the ``reenact eval`` / ``ci`` CLI.
"""

from pathlib import Path

from reenact.evals import (
    Check,
    Scenario,
    answer_contains,
    called_tool,
    no_mutating_tool_reexecuted,
)

SCENARIOS = Path(__file__).resolve().parent / "scenarios"

# A topical keyword present in every good triage of each issue (robust across the
# clean re-record, so it never false-positives).
_ANSWER_KEYWORD = {"42": "password", "57": "billing", "63": "429"}


def _checks_for(issue_id: str) -> list[Check]:
    return [
        called_tool("search_docs"),
        called_tool("label_issue"),
        called_tool("post_reply"),
        no_mutating_tool_reexecuted(),
        answer_contains(_ANSWER_KEYWORD[issue_id]),
    ]


def demo_scenarios(set_name: str) -> list[Scenario]:
    """Build the demo scenarios for a recorded set (e.g. ``baseline``)."""
    base = SCENARIOS / set_name
    return [
        Scenario.from_cassette(
            base / f"issue-{issue_id}.json",
            _checks_for(issue_id),
            name=f"triage-{issue_id}",
        )
        for issue_id in _ANSWER_KEYWORD
    ]
