# Reenact

**Regression testing for LLM agents.** Record an agent run once, replay it
deterministically offline at $0 with no API key, and gate regressions on every pull
request.

Think VCR.py, but for full multi-step agent trajectories: record -> replay ->
evaluate -> gate.

![reenact replays a recorded agent run offline: a model swap turns the gate red, reverting it turns green again](https://raw.githubusercontent.com/Manas-33/Reenact/main/docs/demo.gif)

## Install

```bash
pip install reenact
```

## Quickstart

Wrap the client your agent already uses — reenact records each call and returns the
real response unchanged, so the agent behaves exactly as before:

```python
import reenact
from reenact.store import save_cassette

with reenact.recording(client) as run:   # an Anthropic or OpenAI client
    answer = run_agent(client)            # your agent, unchanged

save_cassette(run.trajectory, "scenarios/run.json")
```

Then scaffold the harness, draft a suite from the recording, and set a baseline:

```bash
reenact init                                                    # scaffold evals/ + a PR workflow
reenact suggest evals/scenarios/run.json -o evals/suite.toml    # draft the checks
reenact eval evals/suite.toml --write-baseline evals/baseline.json
```

On every PR the committed GitHub Action replays the suite offline and blocks the
merge only when a check regresses against the baseline, posting a sticky comment and
a merge-gating check-run.

## What it checks

- **Assertions** — deterministic and free: `called_tool`, `did_not_call_tool`,
  `answer_contains`, `no_mutating_tool_reexecuted` (a mutating tool is never re-run on
  replay), and more.
- **Criteria** — model-judged, evidence-cited yes/no questions for faithfulness and
  grounding (needs an API key at gate time).

Works with the **Anthropic SDK**, **OpenAI SDK**, and **LangGraph**.

## Documentation

Full docs, examples, and the demo repo: **https://github.com/Manas-33/Reenact**

## License

MIT
