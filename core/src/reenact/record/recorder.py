"""Assemble captured LLM calls into a trajectory."""

from typing import Any

from reenact.record.hashing import hash_request
from reenact.record.redaction import DEFAULT_SCRUB_KEYS, redact
from reenact.schema import LLMCallEvent, TokenUsage, Trajectory


class Recorder:
    """Collects captured calls into a single trajectory.

    Provider-agnostic: SDK adapters (Anthropic, OpenAI, ...) convert their
    request and response objects into plain dicts and hand them here. Sensitive
    values are scrubbed before anything is stored.
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        scrub_keys: frozenset[str] = DEFAULT_SCRUB_KEYS,
    ) -> None:
        self.trajectory = Trajectory(name=name)
        self._scrub_keys = scrub_keys

    def record_llm_call(
        self,
        *,
        provider: str,
        model: str,
        request: dict[str, Any],
        response: dict[str, Any],
        usage: TokenUsage | None = None,
        cost_usd: float | None = None,
        latency_ms: float | None = None,
    ) -> LLMCallEvent:
        """Capture one LLM call as an event and append it to the trajectory.

        Request and response are redacted before storage, and the request
        fingerprint is taken over the redacted request so the cassette stays
        self-consistent.
        """
        safe_request: dict[str, Any] = redact(request, self._scrub_keys)
        safe_response: dict[str, Any] = redact(response, self._scrub_keys)
        event = LLMCallEvent(
            seq=len(self.trajectory.events),
            provider=provider,
            model=model,
            request=safe_request,
            response=safe_response,
            request_hash=hash_request(safe_request),
            usage=usage,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )
        self.trajectory.events.append(event)
        return event
