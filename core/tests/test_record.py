"""Recorder captures calls into a trajectory and fingerprints requests."""

from typing import Any

from reenact.record import Recorder, hash_request


def test_request_hash_is_order_independent() -> None:
    a = hash_request({"model": "m", "messages": [], "max_tokens": 10})
    b = hash_request({"max_tokens": 10, "messages": [], "model": "m"})
    assert a == b
    assert a.startswith("sha256:")


def test_request_hash_changes_with_content() -> None:
    a = hash_request({"messages": [{"role": "user", "content": "hi"}]})
    b = hash_request({"messages": [{"role": "user", "content": "bye"}]})
    assert a != b


def test_recorder_builds_a_trajectory() -> None:
    request: dict[str, Any] = {"model": "claude-sonnet-4-5", "messages": []}
    rec = Recorder(name="run")
    event = rec.record_llm_call(
        provider="anthropic",
        model="claude-sonnet-4-5",
        request=request,
        response={"id": "msg_1", "content": []},
    )
    assert event.seq == 0
    assert event.request_hash == hash_request(request)
    assert rec.trajectory.events == [event]


def test_recorder_assigns_monotonic_seq() -> None:
    rec = Recorder()
    first = rec.record_llm_call(provider="p", model="m", request={"n": 1}, response={})
    second = rec.record_llm_call(provider="p", model="m", request={"n": 2}, response={})
    assert (first.seq, second.seq) == (0, 1)
    assert len(rec.trajectory.events) == 2
