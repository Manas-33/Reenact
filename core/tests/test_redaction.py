"""Redaction scrubs sensitive values from nested request/response bodies."""

from typing import Any

from reenact.record import REDACTED, Recorder, redact
from reenact.schema import LLMCallEvent


def test_redacts_sensitive_keys_case_insensitively() -> None:
    data: dict[str, Any] = {
        "Authorization": "Bearer sk-secret",
        "api_key": "sk-123",
        "model": "claude-sonnet-4-5",
    }
    out = redact(data)
    assert out["Authorization"] == REDACTED
    assert out["api_key"] == REDACTED
    assert out["model"] == "claude-sonnet-4-5"


def test_redacts_nested_values() -> None:
    data: dict[str, Any] = {
        "headers": {"x-api-key": "sk-abc"},
        "messages": [{"role": "user", "content": "hi", "secret": "leak"}],
    }
    out = redact(data)
    assert out["headers"]["x-api-key"] == REDACTED
    assert out["messages"][0]["secret"] == REDACTED
    assert out["messages"][0]["content"] == "hi"


def test_redact_does_not_mutate_input() -> None:
    data: dict[str, Any] = {"api_key": "sk-123"}
    redact(data)
    assert data["api_key"] == "sk-123"


def test_recorder_scrubs_before_storing() -> None:
    rec = Recorder()
    rec.record_llm_call(
        provider="anthropic",
        model="m",
        request={"model": "m", "api_key": "sk-secret", "messages": []},
        response={"id": "1", "content": []},
    )
    event = rec.trajectory.events[0]
    assert isinstance(event, LLMCallEvent)
    assert event.request["api_key"] == REDACTED
    assert event.request["model"] == "m"
