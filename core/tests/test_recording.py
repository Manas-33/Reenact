"""The recording() context manager captures calls with no changes to agent code."""

from typing import Any

import pytest

from reenact import recording
from reenact.schema import LLMCallEvent


class _StubMessage:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        return self._data


class _StubMessages:
    def create(self, **kwargs: Any) -> _StubMessage:
        return _StubMessage({"id": "msg_1", "content": [], "echo": kwargs})


class _StubClient:
    def __init__(self) -> None:
        self.messages = _StubMessages()


def _agent(client: Any) -> None:
    # "agent code": a normal call, unaware reenact exists
    client.messages.create(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "hi"}],
    )


def test_records_calls_made_inside_the_block() -> None:
    client = _StubClient()
    with recording(client) as rec:
        _agent(client)
    assert len(rec.trajectory.events) == 1
    event = rec.trajectory.events[0]
    assert isinstance(event, LLMCallEvent)
    assert event.model == "claude-sonnet-4-5"


def test_restores_the_original_method_after_the_block() -> None:
    client = _StubClient()
    with recording(client):
        pass
    assert "create" not in vars(client.messages)


def test_restores_even_when_the_block_raises() -> None:
    client = _StubClient()
    with pytest.raises(ValueError, match="boom"), recording(client):
        raise ValueError("boom")
    assert "create" not in vars(client.messages)
