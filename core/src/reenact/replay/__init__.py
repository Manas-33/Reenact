"""Replay: return recorded responses instead of calling a provider.

The player matches each live call against the recording by (call type, sequence)
and verifies it by request fingerprint, so a replayed run is deterministic and
offline. A call that no longer matches is a divergence - raised in strict mode,
collected in lenient mode - never a silently wrong answer.
"""

from reenact.replay.divergence import Divergence, DivergenceError, DivergenceKind
from reenact.replay.live import replaying
from reenact.replay.player import Player, ReplayMode

__all__ = [
    "Divergence",
    "DivergenceError",
    "DivergenceKind",
    "Player",
    "ReplayMode",
    "replaying",
]
