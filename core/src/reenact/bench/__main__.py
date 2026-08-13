"""Run the benchmarks and write the results file.

Usage: ``python -m reenact.bench``.
"""

import json
from pathlib import Path

from reenact.bench.determinism import measure_determinism
from reenact.bench.overhead import measure_overhead

RESULTS = Path(__file__).resolve().parents[3] / "bench" / "results" / "latest.json"


def main() -> None:
    benchmarks = [measure_overhead()]
    determinism = measure_determinism()
    if determinism["total_runs"] > 0:
        benchmarks.append(determinism)
    results = {"benchmarks": benchmarks}
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
