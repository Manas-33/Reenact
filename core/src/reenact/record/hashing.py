"""Canonical request hashing - the fingerprint replay matches calls on."""

import hashlib
import json
from typing import Any


def hash_request(request: dict[str, Any]) -> str:
    """Return a stable ``sha256:`` fingerprint of a request body.

    Keys are sorted, so the hash is independent of dict ordering; the same
    logical request always produces the same fingerprint, on any machine.
    """
    canonical = json.dumps(
        request, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
