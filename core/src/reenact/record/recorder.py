"""Assemble captured LLM calls into a trajectory."""

from typing import Any

from reenact.record.hashing import hash_request
from reenact.schema import LLMCallEvent, TokenUsage, Trajectory


class Recorder:
    """Collects captured calls into a single trajectory.

    Provider-agnostic: SDK adapters (Anthropic, OpenAI, ...) convert their
    request and response objects into plain dicts and hand them here.
    """

    def __init__(self, name: str | None = None) -> None:
        self.trajectory = Trajectory(name=name)

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
        """Capture one LLM call as an event and append it to the trajectory."""
        event = LLMCallEvent(
            seq=len(self.trajectory.events),
            provider=provider,
            model=model,
            request=request,
            response=response,
            request_hash=hash_request(request),
            usage=usage,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )
        self.trajectory.events.append(event)
        return event
