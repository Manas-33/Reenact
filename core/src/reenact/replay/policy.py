"""The replay side-effect policy: which tools may re-run, which are substituted.

The safe default is to substitute everything - hand back the recorded result and
never touch the real tool - so a replay can never fire a side effect. A tool is
re-executed live only when it is classified ``READ_ONLY`` *and* re-execution is
explicitly opted into. This is the guarantee behind "a mutating tool is never
re-fired on replay."
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from reenact.schema import SideEffect


def _no_overrides() -> dict[str, SideEffect]:
    """Typed empty-mapping factory (a bare ``dict`` factory reads as Unknown)."""
    return {}


@dataclass(frozen=True)
class ReplayPolicy:
    """Decides, per tool, whether replay substitutes the recorded result or
    re-runs the real tool live.

    ``overrides`` reclassify a tool by name, winning over the class recorded on
    the event; a tool recorded (and not overridden) as ``UNKNOWN`` is treated as
    ``default``. Re-execution happens only for a ``READ_ONLY`` tool when
    ``reexecute_read_only`` is set; everything else is substituted.
    """

    overrides: dict[str, SideEffect] = field(default_factory=_no_overrides)
    default: SideEffect = SideEffect.MUTATING
    reexecute_read_only: bool = False

    def classify(self, name: str, recorded: SideEffect) -> SideEffect:
        """Resolve the effective side-effect class for a tool call."""
        effect = self.overrides.get(name, recorded)
        return self.default if effect is SideEffect.UNKNOWN else effect

    def should_substitute(self, name: str, recorded: SideEffect) -> bool:
        """True if the recorded result is substituted (the real tool not run)."""
        effect = self.classify(name, recorded)
        return not (effect is SideEffect.READ_ONLY and self.reexecute_read_only)

    @classmethod
    def from_config(
        cls,
        overrides: Mapping[str, str] | None = None,
        *,
        default: str = SideEffect.MUTATING.value,
        reexecute_read_only: bool = False,
    ) -> ReplayPolicy:
        """Build a policy from plain strings - the JSON/TOML-friendly config form."""
        resolved = {
            name: SideEffect(value) for name, value in (overrides or {}).items()
        }
        return cls(
            overrides=resolved,
            default=SideEffect(default),
            reexecute_read_only=reexecute_read_only,
        )
