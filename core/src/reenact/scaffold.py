"""Scaffold a project onto the reenact gate: the `reenact init` files.

`reenact init` writes the boring parts of the on-ramp so a new project does not
start from a blank page: a `record.py` template (wrap the agent in two lines), a
`suite.toml` skeleton (filled in by `reenact suggest`), and a pull-request workflow
that runs the gate. The user fills in only the one thing reenact cannot know - how
to run their agent.

Writing is safe by default: an existing file is skipped, never overwritten, unless
``force`` is set. The templates are plain strings (no model, no network), so the
scaffold is deterministic and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

RECORD_TEMPLATE = '''\
"""Record one run of your agent into a committed cassette (a test fixture).

Reenact just observes: wrap your normal agent call in ``reenact.recording(client)``
and it captures every model and tool call, returning the real responses unchanged.
Record once (this needs your API key); every replay after is offline and free.

Fill in the two TODOs below, then:

    python evals/record.py                                        # writes a cassette
    reenact suggest evals/scenarios/run.json -o evals/suite.toml  # draft the checks
"""

from pathlib import Path

import reenact
from reenact.store import save_cassette


def main() -> None:
    # TODO 1: create your model client (Anthropic or OpenAI), as you already do.
    #     import anthropic
    #     client = anthropic.Anthropic()
    client = ...  # <- your model client

    with reenact.recording(client) as recorder:
        # TODO 2: run your agent using `client`, exactly as in production.
        #     run_my_agent(client)
        ...

    output = Path(__file__).parent / "scenarios" / "run.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    save_cassette(recorder.trajectory, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
'''

SUITE_TEMPLATE = """\
# Reenact eval suite. Generate it from a recording instead of writing it by hand:
#
#   reenact suggest evals/scenarios/run.json -o evals/suite.toml
#
# That fills this file with a [[scenario]] and checks derived from the run (which
# tools the agent called, the mutating-tool safety check, an answer keyword). Then
# record the baseline the CI gate compares against:
#
#   reenact eval evals/suite.toml --write-baseline evals/baseline.json
#
# The shape, for reference:
#
#   [[scenario]]
#   name = "my-scenario"
#   cassette = "scenarios/run.json"
#
#     [[scenario.check]]
#     type = "called_tool"
#     name = "my_tool"
"""

WORKFLOW_TEMPLATE = """\
# Gates pull requests with Reenact: replays the committed scenario suite offline
# ($0, no key) and fails the check only on a regression versus the baseline, then
# posts a sticky PR comment and a merge-gating check-run.
name: Reenact
on: pull_request

permissions:
  contents: read
  pull-requests: write  # post/update the PR comment
  checks: write         # open the check-run

jobs:
  gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      # TODO: replace `your-org` with the published reenact action ref (pin a tag).
      - uses: your-org/reenact/action@v1
        with:
          suite: evals/suite.toml
          baseline: evals/baseline.json
          token: ${{ github.token }}
"""

# Relative path -> file contents. Ordered so `evals/` files come before the workflow.
_FILES: dict[str, str] = {
    "evals/record.py": RECORD_TEMPLATE,
    "evals/suite.toml": SUITE_TEMPLATE,
    ".github/workflows/reenact.yml": WORKFLOW_TEMPLATE,
}


@dataclass(frozen=True)
class ScaffoldResult:
    """One scaffolded file: where it went, and whether it was written or skipped."""

    path: Path
    written: bool


def scaffold(target: Path, *, force: bool = False) -> list[ScaffoldResult]:
    """Write the scaffold files under ``target``, skipping any that already exist.

    An existing file is left untouched (``written=False``) unless ``force`` is set,
    so re-running ``init`` never clobbers edits. Parent directories are created as
    needed. Returns one :class:`ScaffoldResult` per file, in a stable order.
    """
    results: list[ScaffoldResult] = []
    for relative, content in _FILES.items():
        path = target / relative
        if path.exists() and not force:
            results.append(ScaffoldResult(path=path, written=False))
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        results.append(ScaffoldResult(path=path, written=True))
    return results
