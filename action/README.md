# `reenact` GitHub Action

Composite Action that wraps the Reenact CLI to run the regression gate on pull
requests: it replays the recorded scenario suite, evaluates it against a
baseline, and posts a sticky PR comment with the regression diff plus a
check-run.
