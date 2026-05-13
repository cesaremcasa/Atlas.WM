"""Compute a stable hash of the current environment (AD-6).

The env_hash is embedded in every checkpoint's metadata so that any model loaded
in a different environment can be flagged as potentially non-reproducible.

Hash is SHA-256 of requirements.lock (the fully-pinned, hash-verified lockfile).
Only the first 16 hex chars (64 bits) are used to keep the string compact.
"""

from __future__ import annotations

import hashlib
import os


def compute_env_hash(lock_path: str | None = None) -> str:
    """Return a 16-char hex digest identifying the current pinned environment.

    Falls back to 'unknown-no-lockfile' if requirements.lock is not found.

    Args:
        lock_path: Override for the lockfile path. If None, searches for
                   requirements.lock relative to the current working directory
                   and up to 3 parent directories.
    """
    if lock_path is not None:
        return _hash_file(lock_path)

    candidates = [
        "requirements.lock",
        "../requirements.lock",
        "../../requirements.lock",
        "../../../requirements.lock",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return _hash_file(candidate)

    return "unknown-no-lockfile"


def _hash_file(path: str) -> str:
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return digest[:16]
