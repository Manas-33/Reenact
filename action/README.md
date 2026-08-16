# `reenact` GitHub Action

Composite Action that wraps the Reenact CLI to run the regression gate on pull
requests. It replays the committed scenario suite **offline** ($0, no network, no
API key), diffs the result against a committed baseline, and fails the check only on
a *regression* - then posts a sticky PR comment and a merge-gating check-run.

## Usage

```yaml
name: Reenact
on: pull_request

permissions:
  contents: read
  pull-requests: write   # to post/update the PR comment
  checks: write          # to open the check-run

jobs:
  gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: your-org/reenact/action@v1
        with:
          suite: evals/suite.toml
          baseline: evals/baseline.json
          token: ${{ github.token }}
```

The gate goes **red** only when a check that passed in the baseline now fails (or a
judge score drops past the tolerance). A brand-new failing check is reported but does
not block; an *advisory* criterion regresses as a warning, never a block.

## Inputs

| Input | Required | Default | Description |
| --- | --- | --- | --- |
| `suite` | yes | - | Path to the TOML eval suite (relative to `working-directory`). |
| `baseline` | yes | - | Path to the committed baseline JSON to diff against. |
| `tolerance` | no | `0.05` | Judge-score drop tolerated before it counts as a regression. |
| `working-directory` | no | `.` | Directory to run `reenact` in. |
| `python-version` | no | `3.12` | Python version to set up. |
| `version` | no | `reenact` | `pip install` spec: the PyPI name, a pin (`reenact==0.1.0`), a `git+https://...` URL, or a local path (`./core`). |
| `token` | no | `""` | GitHub token for posting the comment + check-run (`${{ github.token }}`). Omit to skip posting; the job still passes/fails on the gate. |

## Permissions

Posting needs a token with `pull-requests: write` (the sticky comment) and
`checks: write` (the check-run). Grant them in the workflow's `permissions:` block as
shown above. With no `token`, the gate still runs and sets the job's pass/fail - it
just does not comment.
