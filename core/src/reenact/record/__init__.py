"""Recording: capture LLM (and later tool) calls into a trajectory.

The recorder core is provider-agnostic; SDK adapters (Anthropic, OpenAI, ...)
turn their request and response objects into plain dicts and feed it. Sensitive
values are scrubbed at the boundary before anything is stored.
"""

from reenact.record.hashing import hash_request
from reenact.record.recorder import Recorder
from reenact.record.redaction import DEFAULT_SCRUB_KEYS, REDACTED, redact

__all__ = [
    "DEFAULT_SCRUB_KEYS",
    "REDACTED",
    "Recorder",
    "hash_request",
    "redact",
]
