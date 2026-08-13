"""Nondeterminism shims: byte-identical replay despite a clock, RNG, or stream."""

from pathlib import Path
from typing import Any

import pytest

from reenact.record import Recorder
from reenact.replay import (
    Clock,
    DivergenceError,
    Player,
    Rng,
    load_clock,
    load_rng,
    reassemble_text,
    replay_stream,
    save_entropy,
)
from reenact.store import load_cassette, save_cassette


def _timestamped_request(clock: Clock) -> dict[str, Any]:
    stamp = clock.now().isoformat()
    return {"model": "m", "messages": [{"role": "user", "content": f"now={stamp}"}]}


def _recorded_timestamped_run() -> Recorder:
    clock = Clock()
    rec = Recorder()
    rec.record_llm_call(
        provider="anthropic",
        model="m",
        request=_timestamped_request(clock),
        response={"ok": True},
    )
    save_entropy(rec.trajectory, clock=clock)
    return rec


# --- clock injection ---


def test_clock_replay_makes_a_timestamped_request_byte_identical() -> None:
    rec = _recorded_timestamped_run()
    # The frozen clock reproduces the recorded timestamp -> identical request.
    player = Player(rec.trajectory)
    replayed = _timestamped_request(load_clock(rec.trajectory))
    assert player.replay_llm_call(replayed) == {"ok": True}  # no divergence


def test_without_the_clock_shim_the_same_agent_diverges() -> None:
    rec = _recorded_timestamped_run()
    player = Player(rec.trajectory)
    # A different clock stamps a different time -> the fingerprint drifts.
    with pytest.raises(DivergenceError):
        player.replay_llm_call(_timestamped_request(Clock.replaying([0.0])))


def test_clock_reads_past_the_log_diverge() -> None:
    clock = Clock()
    clock.time()  # one recorded read
    replay = Clock.replaying(clock.log)
    replay.time()  # ok - matches the log
    with pytest.raises(DivergenceError):
        replay.time()  # past the end of the log


# --- rng injection ---


def test_rng_replays_the_same_sequence_from_the_seed() -> None:
    rng = Rng()
    recorded = [rng.random() for _ in range(5)]
    replay = Rng(seed=rng.seed)
    assert [replay.random() for _ in range(5)] == recorded


def test_rng_token_is_reproducible() -> None:
    rng = Rng()
    recorded = rng.token()
    assert Rng(seed=rng.seed).token() == recorded


# --- persistence ---


def test_entropy_round_trips_through_a_cassette(tmp_path: Path) -> None:
    clock = Clock()
    clock.now()
    clock.now()
    rng = Rng()
    rng.random()
    rec = Recorder()
    rec.record_llm_call(provider="anthropic", model="m", request={"a": 1}, response={})
    save_entropy(rec.trajectory, clock=clock, rng=rng)

    path = tmp_path / "run.json"
    save_cassette(rec.trajectory, path)
    loaded = load_cassette(path)

    assert load_clock(loaded).log == clock.log
    assert load_rng(loaded).seed == rng.seed


# --- streaming reassembly ---


def test_replay_stream_re_emits_recorded_chunks_offline() -> None:
    chunks: list[Any] = [
        {"delta": {"text": "Hel"}},
        {"delta": {"text": "lo"}},
        {"other": "ignored"},
    ]
    emitted = [chunk.model_dump() for chunk in replay_stream(chunks)]
    assert emitted == chunks


def test_reassemble_text_folds_the_deltas() -> None:
    chunks: list[Any] = [
        {"delta": {"text": "Hel"}},
        {"delta": {"text": "lo"}},
        {"type": "message_stop"},
    ]
    assert reassemble_text(chunks) == "Hello"
