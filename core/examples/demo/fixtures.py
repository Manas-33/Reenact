"""The demo's fixture world: a small docs corpus and an issue tracker.

The agent's tools read and write this fixture, not a live GitHub - a demo must
never post real comments, and substituting the mutating tools on replay is the
guarantee being shown off. Fixed data keeps recordings deterministic.
"""

# The product docs the read-only tools search and read.
DOCS: dict[str, str] = {
    "auth.md": (
        "Password reset: open Settings > Security and click Reset Password. "
        "A reset link is emailed and expires in one hour. If the link errors, "
        "it has usually expired - request a fresh one."
    ),
    "billing.md": (
        "Billing: invoices are issued monthly. A duplicate charge is refunded "
        "within five business days to the original payment method. Ask the user "
        "for the invoice id before escalating."
    ),
    "api.md": (
        "API: the rate limit is 100 requests per minute per key. A 429 response "
        "means the limit was exceeded; back off and retry with exponential delay."
    ),
}

# The issues the agent triages, keyed by id.
ISSUES: dict[str, dict[str, str]] = {
    "42": {
        "title": "Password reset link doesn't work",
        "body": "I click the reset link in my email but it just shows an error page.",
    },
    "57": {
        "title": "Charged twice this month",
        "body": "My card was billed two times for a single subscription this month.",
    },
    "63": {
        "title": "Getting 429 errors from the API",
        "body": "All of a sudden every request to the API fails with a 429.",
    },
}


def issue_text(issue_id: str) -> str:
    """Render an issue as the prompt the agent triages."""
    issue = ISSUES[issue_id]
    return f"Issue #{issue_id}: {issue['title']}\n\n{issue['body']}"
