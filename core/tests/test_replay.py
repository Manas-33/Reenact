"""Replay hands back recorded responses offline, byte-identical, or flags drift."""

from pathlib import Path
from typing import Any

import pytest

from reenact.record import Recorder
from reenact.replay import DivergenceError, Player
from reenact.store import load_cassette, save_cassette

REQUEST: dict[str, Any] = {
    "model": "claude-sonnet-4-5",
    "messages": [{"role": "user", "content": "hi"}],
}
RESPONSE: dict[str, Any] = {
    "id": "msg_1",
    "content": [{"type": "text", "text": "Hello!"}],
}


def _record_and_save(tmp_path: Path) -> Path:
    rec = Recorder(name="run")
    rec.record_llm_call(
        provider="anthropic",
        model="claude-sonnet-4-5",
        request=REQUEST,
        response=RESPONSE,
    )
    path = tmp_path / "run.json"
    save_cassette(rec.trajectory, path)
    return path


def test_replay_returns_recorded_response_offline(tmp_path: Path) -> None:
    # record -> save cassette -> load -> replay, with no network call anywhere
    player = Player(load_cassette(_record_and_save(tmp_path)))
    assert player.replay_llm_call(REQUEST) == RESPONSE


def test_replay_flags_divergence_on_a_changed_request(tmp_path: Path) -> None:
    player = Player(load_cassette(_record_and_save(tmp_path)))
    changed: dict[str, Any] = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "different"}],
    }
    with pytest.raises(DivergenceError):
        player.replay_llm_call(changed)


def test_replay_stops_after_the_recorded_calls(tmp_path: Path) -> None:
    player = Player(load_cassette(_record_and_save(tmp_path)))
    assert player.replay_llm_call(REQUEST) == RESPONSE
    with pytest.raises(DivergenceError):
        player.replay_llm_call(REQUEST)
