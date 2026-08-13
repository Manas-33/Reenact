"""The overhead benchmark produces a sane, denominated result."""

from reenact.bench.overhead import measure_overhead


def test_measure_overhead_returns_a_denominated_result() -> None:
    result = measure_overhead(iterations=50, warmup=10)
    assert result["iterations"] == 50
    assert isinstance(result["ms_per_call"], float)
    assert result["ms_per_call"] > 0
    assert result["within_floor"] is (result["ms_per_call"] < result["floor_ms"])
