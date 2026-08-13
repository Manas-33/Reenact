"""The OpenAI adapter records chat completions without depending on the SDK."""

from typing import Any

import pytest

import reenact
from reenact.record import Recorder
from reenact.record.openai import record_completion
from reenact.schema import LLMCallEvent, TokenUsage


class _StubCompletion:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        return self._data


class _StubCompletions:
    def __init__(self, response: dict[str, Any]) -> None:
        self._response = response

    def create(self, **kwargs: Any) -> _StubCompletion:
        return _StubCompletion(self._response)


class _StubChat:
    def __init__(self, response: dict[str, Any]) -> None:
        self.completions = _StubCompletions(response)


class _StubClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.chat = _StubChat(response)


_RESPONSE: dict[str, Any] = {
    "id": "chatcmpl-1",
    "model": "gpt-4o",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi!"}}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
}


def test_record_completion_maps_openai_usage() -> None:
    rec = Recorder()
    record_completion(
        _StubClient(_RESPONSE),
        rec,
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
    )
    event = rec.trajectory.events[0]
    assert isinstance(event, LLMCallEvent)
    assert event.provider == "openai"
    assert event.model == "gpt-4o"
    # OpenAI's prompt/completion tokens map onto the shared TokenUsage
    assert event.usage == TokenUsage(input_tokens=12, output_tokens=3)


def test_reenact_recording_dispatches_to_openai() -> None:
    client = _StubClient(_RESPONSE)
    with reenact.recording(client) as rec:
        client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
        )
    event = rec.trajectory.events[0]
    assert isinstance(event, LLMCallEvent)
    assert event.provider == "openai"


def test_reenact_recording_rejects_unknown_client() -> None:
    with pytest.raises(TypeError):
        reenact.recording(object())
