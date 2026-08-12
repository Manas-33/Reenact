"""The Anthropic adapter records a live call without depending on the SDK."""

from typing import Any

from reenact.record import Recorder
from reenact.record.anthropic import record_message
from reenact.schema import LLMCallEvent, TokenUsage


class _StubMessage:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        return self._data


class _StubMessages:
    def __init__(self, response: dict[str, Any]) -> None:
        self._response = response
        self.received: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> _StubMessage:
        self.received = kwargs
        return _StubMessage(self._response)


class _StubClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.messages = _StubMessages(response)


def test_records_and_returns_the_response() -> None:
    response: dict[str, Any] = {
        "id": "msg_1",
        "model": "claude-sonnet-4-5",
        "content": [{"type": "text", "text": "Hello!"}],
        "usage": {"input_tokens": 10, "output_tokens": 3},
    }
    client = _StubClient(response)
    rec = Recorder()

    returned = record_message(
        client,
        rec,
        model="claude-sonnet-4-5",
        max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
    )

    # the caller still gets the real response object back, unchanged
    assert returned.model_dump() == response

    # and the call was captured into the trajectory
    event = rec.trajectory.events[0]
    assert isinstance(event, LLMCallEvent)
    assert event.provider == "anthropic"
    assert event.model == "claude-sonnet-4-5"
    assert event.response == response
    assert event.usage == TokenUsage(input_tokens=10, output_tokens=3)
    assert event.request["max_tokens"] == 100
    assert client.messages.received["model"] == "claude-sonnet-4-5"
