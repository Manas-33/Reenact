"""Replay: return recorded responses instead of calling a provider.

The player matches each live call against the recording by sequence and verifies
it by request fingerprint, so a replayed run is deterministic and offline. A
call that no longer matches raises a divergence rather than being hidden.
"""

from reenact.replay.live import replaying
from reenact.replay.player import DivergenceError, Player

__all__ = ["DivergenceError", "Player", "replaying"]
