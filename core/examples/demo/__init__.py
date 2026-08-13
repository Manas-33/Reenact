"""The issue-triage demo agent - the seed of the public reenact-demo repo.

A real LangGraph agent that triages a support issue: search the docs, label the
issue, post a reply, summarize. Its read-only tools (``search_docs`` /
``read_file``) may re-run on replay; its mutating tools (``label_issue`` /
``post_reply``) are always substituted. Recorded once against the live model,
replayed and gated forever offline.
"""
