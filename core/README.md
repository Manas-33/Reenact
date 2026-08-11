# Reenact

**Regression testing for LLM agents.** Record an agent run once, replay it
deterministically offline at $0, and gate regressions in CI.

Think VCR.py, but for full multi-step agent trajectories: record -> replay ->
evaluate.