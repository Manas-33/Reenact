"""Measure the per-call overhead the recorder adds."""

import time
from typing import Any

from reenact.record import Recorder
from reenact.schema import TokenUsage


def _sample_request() -> dict[str, Any]:
    return {
        "model": "claude-sonnet-4-5",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Summarize the passage below in one line."},
            {"role": "assistant", "content": "Sure - here is a one-line summary."},
            {"role": "user", "content": "Now translate that summary into French."},
        ],
    }


def _sample_response() -> dict[str, Any]:
    return {
        "id": "msg_01ABCDEF",
        "model": "claude-sonnet-4-5",
        "role": "assistant",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "Voici la traduction en une ligne."}],
        "usage": {"input_tokens": 320, "output_tokens": 128},
    }


def measure_overhead(iterations: int = 2000, warmup: int = 200) -> dict[str, Any]:
    """Time the recorder's per-call work over a representative LLM call.

    Measures the core recording overhead - redaction, request fingerprinting,
    event construction - excluding provider-specific response serialization.
    """
    request = _sample_request()
    response = _sample_response()
    usage = TokenUsage(input_tokens=320, output_tokens=128)
    recorder = Recorder()

    def _one() -> None:
        recorder.record_llm_call(
            provider="anthropic",
            model="claude-sonnet-4-5",
            request=request,
            response=response,
            usage=usage,
        )

    for _ in range(warmup):
        _one()

    start = time.perf_counter()
    for _ in range(iterations):
        _one()
    elapsed = time.perf_counter() - start

    ms_per_call = elapsed / iterations * 1000.0
    floor_ms = 10.0
    return {
        "metric": "record_overhead_ms_per_call",
        "iterations": iterations,
        "ms_per_call": round(ms_per_call, 4),
        "floor_ms": floor_ms,
        "within_floor": ms_per_call < floor_ms,
        "note": "core recording overhead; excludes provider response serialization",
    }
