"""Recording: capture LLM (and later tool) calls into a trajectory.

The recorder core is provider-agnostic; SDK adapters (Anthropic, OpenAI, ...)
turn their request and response objects into plain dicts and feed it.
"""

from reenact.record.hashing import hash_request
from reenact.record.recorder import Recorder

__all__ = ["Recorder", "hash_request"]
