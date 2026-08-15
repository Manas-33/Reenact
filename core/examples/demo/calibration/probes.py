"""Deliberately-borderline triage trajectories - the calibration stress set.

Hand-authored (synthetic) runs where exactly one criterion is a genuine 50/50
call, so the labelling grid is guaranteed some hard cases a careful human and the
evaluator can honestly disagree on. Calibration over an all-easy grid returns a
near-100% agreement that proves nothing (the ">92% is suspicious" rule); these
probes give it something to bite on.

They are kept separate from the real recorded corpus on purpose: the headline
agreement number stays over the real runs, and these are a labelled *stress set*
reported beside it - synthetic, and honest about being synthetic. Each probe names
the criterion it is meant to make ambiguous.
"""

from reenact.record import hash_request, redact
from reenact.schema import LLMCallEvent, SideEffect, ToolCallEvent, Trajectory


def _triage(
    name: str,
    *,
    question: str,
    doc: str,
    issue_id: str,
    label: str,
    reply: str,
    summary: str,
) -> Trajectory:
    """Assemble a standard think -> search -> label + reply -> summarize run."""

    def llm(seq: int, text: str) -> LLMCallEvent:
        request = {"messages": [{"role": "user", "content": question}], "step": seq}
        return LLMCallEvent(
            seq=seq,
            provider="anthropic",
            model="probe",
            request=request,
            response={"content": [{"type": "text", "text": text}]},
            request_hash=hash_request(redact(request)),
        )

    events = [
        llm(0, "I'll search the documentation and then triage this issue."),
        ToolCallEvent(
            seq=1,
            parent_seq=0,
            name="search_docs",
            arguments={"query": question},
            result=doc,
            side_effect=SideEffect.READ_ONLY,
        ),
        llm(2, "Now I'll label the issue and post a reply."),
        ToolCallEvent(
            seq=3,
            parent_seq=2,
            name="label_issue",
            arguments={"issue_id": issue_id, "label": label},
            result=f"labeled issue #{issue_id} as {label}",
            side_effect=SideEffect.MUTATING,
        ),
        ToolCallEvent(
            seq=4,
            parent_seq=2,
            name="post_reply",
            arguments={"issue_id": issue_id, "body": reply},
            result=f"posted reply to issue #{issue_id}",
            side_effect=SideEffect.MUTATING,
        ),
        llm(5, summary),
    ]
    return Trajectory(name=name, events=events)


def ungrounded_reply() -> Trajectory:
    """Borderline `reply_grounded`: the reply is mostly grounded but adds one claim
    (mobile-app reset) that the retrieved doc never states."""
    return _triage(
        "probe-ungrounded-reply",
        question="My password reset link shows an error page. How do I fix it?",
        doc=(
            "Password reset: open Settings > Security and click Reset Password. A "
            "reset link is emailed and expires in one hour."
        ),
        issue_id="101",
        label="bug",
        reply=(
            "Reset links expire after one hour, so please request a new one from "
            "Settings > Security. Reset links also work directly from the Account "
            "tab of our mobile app if the web page keeps erroring."
        ),
        summary=(
            "I labeled issue #101 as a bug and replied explaining that the link "
            "expires after an hour and that the mobile app can be used instead."
        ),
    )


def debatable_label() -> Trajectory:
    """Borderline `correct_label`: a duplicate-charge-after-failed-reset issue is
    labelled `bug`, but `billing` is just as defensible, and only one is allowed."""
    return _triage(
        "probe-debatable-label",
        question="I was charged twice right after my password reset failed. Why?",
        doc=(
            "Billing: duplicate charges are automatically reversed within 3-5 "
            "business days. For password problems, see the Security section."
        ),
        issue_id="102",
        label="bug",
        reply=(
            "Duplicate charges are automatically reversed within 3-5 business days, "
            "so the second charge should drop off shortly."
        ),
        summary=(
            "I labeled issue #102 as a bug and replied that the duplicate charge "
            "auto-reverses within a few business days."
        ),
    )


def overstated_summary() -> Trajectory:
    """Borderline `faithful`: the final answer claims it raised the account's rate
    limit, but no such tool was ever called - the summary overstates the run."""
    return _triage(
        "probe-overstated-summary",
        question="Your API keeps returning 429 errors during peak hours.",
        doc=(
            "A 429 means the request was rate limited. Back off and retry with "
            "exponential backoff; limits reset each minute."
        ),
        issue_id="103",
        label="api",
        reply=(
            "A 429 means you're being rate limited. Use exponential backoff and "
            "retry; the limit resets each minute."
        ),
        summary=(
            "I labeled issue #103 as api, replied with backoff guidance, and raised "
            "your account's rate limit to resolve the errors."
        ),
    )


# The criterion each probe is designed to make borderline (for the analysis notes,
# never shown to a rater - they label blind).
PROBE_TARGETS = {
    "probe-ungrounded-reply": "reply_grounded",
    "probe-debatable-label": "correct_label",
    "probe-overstated-summary": "faithful",
}


def probes() -> list[Trajectory]:
    """The stress-set trajectories, in a stable order."""
    return [ungrounded_reply(), debatable_label(), overstated_summary()]
