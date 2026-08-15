"""Load a committable suite config (TOML) into runnable scenarios.

A suite file lists scenarios; each names a cassette and a set of checks, written
as plain tables so the whole thing lives in git next to the recordings it gates.
Check specs are resolved to the Check factories from this package. A ``judge``
check and any ``[[scenario.criterion]]`` additionally need a runtime client,
injected by the caller (the CLI) and never stored in the config - the config
carries only the rubric / question, threshold, and level.
"""

import tomllib
from pathlib import Path
from typing import Any, cast

from reenact.evals.check import (
    Check,
    CriterionLevel,
    answer_contains,
    answer_matches,
    called_tool,
    did_not_call_tool,
    no_mutating_tool_reexecuted,
    replays_clean,
    tool_call_count,
)
from reenact.evals.judge import DEFAULT_THRESHOLD, judged
from reenact.evals.scenario import Scenario
from reenact.evals.structured import Criterion, structured_eval


class SuiteConfigError(ValueError):
    """Raised when a suite config file is missing, malformed, or references an
    unknown check."""


def _table(value: Any, ctx: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SuiteConfigError(f"{ctx} must be a table")
    return cast(dict[str, Any], value)


def _req_str(spec: dict[str, Any], key: str, ctx: str) -> str:
    value = spec.get(key)
    if not isinstance(value, str):
        raise SuiteConfigError(f"{ctx}: '{key}' must be a string")
    return value


def _req_int(spec: dict[str, Any], key: str, ctx: str) -> int:
    value = spec.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SuiteConfigError(f"{ctx}: '{key}' must be an integer")
    return value


def _opt_name(spec: dict[str, Any]) -> str | None:
    value = spec.get("name")
    return value if isinstance(value, str) else None


def _build_check(spec: dict[str, Any], *, judge_client: Any, ctx: str) -> Check:
    """Resolve one check table to a Check, dispatching on its ``type``."""
    kind = spec.get("type")
    if kind == "answer_contains":
        case_sensitive = bool(spec.get("case_sensitive", False))
        return answer_contains(
            _req_str(spec, "value", ctx), case_sensitive=case_sensitive
        )
    if kind == "answer_matches":
        return answer_matches(_req_str(spec, "pattern", ctx))
    if kind == "called_tool":
        return called_tool(_req_str(spec, "name", ctx))
    if kind == "did_not_call_tool":
        return did_not_call_tool(_req_str(spec, "name", ctx))
    if kind == "tool_call_count":
        return tool_call_count(
            _req_str(spec, "name", ctx), _req_int(spec, "count", ctx)
        )
    if kind == "replays_clean":
        return replays_clean()
    if kind == "no_mutating_tool_reexecuted":
        return no_mutating_tool_reexecuted()
    if kind == "judge":
        if judge_client is None:
            raise SuiteConfigError(
                f"{ctx}: a 'judge' check needs a judge client, but none is "
                "available (set ANTHROPIC_API_KEY and install the anthropic SDK)"
            )
        threshold = spec.get("threshold", DEFAULT_THRESHOLD)
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise SuiteConfigError(f"{ctx}: 'threshold' must be a number")
        return judged(
            judge_client,
            _req_str(spec, "rubric", ctx),
            threshold=float(threshold),
            name=_opt_name(spec),
        )
    raise SuiteConfigError(f"{ctx}: unknown check type {kind!r}")


def _build_criterion(spec: dict[str, Any], ctx: str) -> Criterion:
    """Resolve one ``[[scenario.criterion]]`` table to a :class:`Criterion`."""
    level_raw = spec.get("level", CriterionLevel.BLOCKING.value)
    if not isinstance(level_raw, str):
        raise SuiteConfigError(f"{ctx}: 'level' must be a string")
    try:
        level = CriterionLevel(level_raw)
    except ValueError as exc:
        raise SuiteConfigError(
            f"{ctx}: 'level' must be 'blocking' or 'advisory'"
        ) from exc
    return Criterion(
        id=_req_str(spec, "id", ctx),
        question=_req_str(spec, "question", ctx),
        level=level,
    )


def _criterion_checks(
    table: dict[str, Any], *, judge_client: Any, ctx: str
) -> list[Check]:
    """Build the structured-evaluator checks for a scenario's criteria (if any)."""
    raw_criteria = table.get("criterion", [])
    if not isinstance(raw_criteria, list):
        raise SuiteConfigError(f"{ctx}: 'criterion' must be an array of tables")
    criteria = [
        _build_criterion(_table(raw, f"{ctx}.criterion[{i}]"), f"{ctx}.criterion[{i}]")
        for i, raw in enumerate(cast(list[Any], raw_criteria))
    ]
    if not criteria:
        return []
    if judge_client is None:
        raise SuiteConfigError(
            f"{ctx}: a [[scenario.criterion]] needs a judge client, but none is "
            "available (set ANTHROPIC_API_KEY and install the anthropic SDK)"
        )
    return structured_eval(judge_client, criteria)


def load_suite(path: str | Path, *, judge_client: Any = None) -> list[Scenario]:
    """Load a TOML suite file into a list of runnable scenarios.

    Cassette paths are resolved relative to the suite file's directory, so a
    committed suite is portable. Judge checks need ``judge_client``; assertion
    checks need nothing and run offline.
    """
    source = Path(path)
    try:
        with source.open("rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise SuiteConfigError(f"suite file not found: {source}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise SuiteConfigError(f"invalid TOML in {source}: {exc}") from exc

    raw_scenarios = data.get("scenario")
    if not isinstance(raw_scenarios, list):
        raise SuiteConfigError("suite must define at least one [[scenario]] table")

    base = source.parent
    scenarios: list[Scenario] = []
    for index, raw in enumerate(cast(list[Any], raw_scenarios)):
        ctx = f"scenario[{index}]"
        table = _table(raw, ctx)
        cassette = base / _req_str(table, "cassette", ctx)
        raw_checks = table.get("check", [])
        if not isinstance(raw_checks, list):
            raise SuiteConfigError(f"{ctx}: 'check' must be an array of tables")
        checks = [
            _build_check(
                _table(raw_check, f"{ctx}.check[{i}]"),
                judge_client=judge_client,
                ctx=f"{ctx}.check[{i}]",
            )
            for i, raw_check in enumerate(cast(list[Any], raw_checks))
        ]
        checks.extend(_criterion_checks(table, judge_client=judge_client, ctx=ctx))
        scenarios.append(
            Scenario.from_cassette(cassette, checks, name=_opt_name(table))
        )
    return scenarios
