"""Recording: capture LLM (and later tool) calls into a trajectory.

The recorder core is provider-agnostic; SDK adapters (Anthropic, OpenAI, ...)
turn their request and response objects into plain dicts and feed it. Sensitive
values are scrubbed at the boundary before anything is stored.
"""

from contextlib import AbstractContextManager
from typing import Any

from reenact.record.anthropic import recording as _anthropic_recording
from reenact.record.hashing import hash_request
from reenact.record.openai import recording as _openai_recording
from reenact.record.recorder import Recorder
from reenact.record.redaction import DEFAULT_SCRUB_KEYS, REDACTED, redact


def recording(client: Any) -> AbstractContextManager[Recorder]:
    """Record every LLM call made through ``client`` inside a ``with`` block.

    Detects the client: an Anthropic client (has ``messages``) or an OpenAI
    client (has ``chat``). Agent code inside the block needs no changes.
    """
    if hasattr(client, "messages"):
        return _anthropic_recording(client)
    if hasattr(client, "chat"):
        return _openai_recording(client)
    raise TypeError(
        "reenact.recording expects an Anthropic or OpenAI client; got "
        f"{type(client).__name__}"
    )


__all__ = [
    "DEFAULT_SCRUB_KEYS",
    "REDACTED",
    "Recorder",
    "hash_request",
    "recording",
    "redact",
]
