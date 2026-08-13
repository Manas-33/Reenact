"""Reenact: regression testing for LLM agents.

Record an agent run once, replay it deterministically offline, and gate
regressions in CI.
"""

from reenact.record import recording

__version__ = "0.0.1"

__all__ = ["__version__", "recording"]
