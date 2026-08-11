# Reenact

**Regression testing for LLM agents.** Record an agent run once, replay it
deterministically offline at $0, and fail a PR in CI when the agent gets worse.

Think VCR.py, but for full multi-step agent trajectories: record -> replay ->
evaluate -> gate.

## Layout

| Path       | What it is                                                |
|------------|-----------------------------------------------------------|
| `core/`    | The `reenact` Python package (record, replay, evals, CLI) |
| `action/`  | Composite GitHub Action wrapping the CLI                  |
| `console/` | Hosted viewer and debugger (Next.js)                      |
