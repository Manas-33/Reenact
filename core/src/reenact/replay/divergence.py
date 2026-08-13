"""Structured divergence - what replay reports when a run drifts from its recording.

A divergence is the same value in both replay modes: strict raises it, lenient
collects it. Keeping it a serializable model (not a bare string) is deliberate -
the CI regression gate and the determinism bench both need to report drift
without re-deriving it.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class DivergenceKind(StrEnum):
    """Why a live call failed to match the recording."""

    LLM_REQUEST = "llm_request"  # an LLM request's fingerprint changed
    TOOL_CALL = "tool_call"  # a tool call's name or arguments changed
    EXHAUSTED = "exhausted"  # no recorded call of this type is left to match


class Divergence(BaseModel):
    """One point where a replayed run departed from its recording.

    Carries where it happened (``seq``, ``None`` when the recording is exhausted),
    what kind of drift, the expected vs. actual fingerprints, and a human-readable
    ``message``.
    """

    model_config = ConfigDict(extra="forbid")

    kind: DivergenceKind
    message: str
    seq: int | None = None
    expected: str | None = None
    actual: str | None = None


class DivergenceError(Exception):
    """Raised in strict replay when a live call does not match the recording.

    The attached ``divergence`` is the same value lenient mode collects, so both
    modes describe drift identically - one raises it, the other records it.
    """

    def __init__(self, divergence: Divergence) -> None:
        super().__init__(divergence.message)
        self.divergence: Divergence = divergence
