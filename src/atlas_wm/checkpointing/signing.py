"""HMAC-SHA256 checkpoint signing and manifest verification (AD-4, Block 4).

Security contract:
- Signing key lives in ATLAS_SIGNING_KEY env var (hex-encoded 32 bytes). Never committed.
- manifest.sig is committed to the repo; the key is not.
- Never edit manifest.sig by hand. Always regenerate via scripts/sign_checkpoint.py.

Manifest format (JSON):
{
    "files": {"best_model.safetensors": "<hmac_hex>", ...},
    "manifest_hmac": "<hmac_of_serialized_files_dict>"
}
manifest_hmac authenticates the files dict itself, preventing tampering.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass
class ManifestMismatch:
    path: str
    reason: str


def _hmac_file(path: str, key: bytes) -> str:
    """Return HMAC-SHA256 hex digest of a file's contents."""
    h = hmac.new(key, digestmod=hashlib.sha256)
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _serialize_files(files: dict[str, str]) -> bytes:
    """Stable JSON serialization of the files dict for HMAC computation."""
    return json.dumps(files, sort_keys=True, separators=(",", ":")).encode()


def _hmac_bytes(data: bytes, key: bytes) -> str:
    return hmac.new(key, data, hashlib.sha256).hexdigest()


def build_manifest(checkpoint_dir: str, key: bytes) -> dict[str, Any]:
    """Build a signed manifest for all .safetensors files in checkpoint_dir.

    Returns the manifest dict (not yet written to disk).
    """
    files: dict[str, str] = {}
    for name in sorted(os.listdir(checkpoint_dir)):
        if name.endswith(".safetensors"):
            full_path = os.path.join(checkpoint_dir, name)
            files[name] = _hmac_file(full_path, key)

    manifest_hmac = _hmac_bytes(_serialize_files(files), key)
    return {"files": files, "manifest_hmac": manifest_hmac}


def write_manifest(manifest_path: str, manifest: dict[str, Any]) -> None:
    """Write manifest dict to manifest_path as JSON."""
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def load_manifest(manifest_path: str) -> dict[str, Any]:
    """Load and return the raw manifest dict from disk."""
    with open(manifest_path) as f:
        return json.load(f)


def verify_manifest(manifest_path: str, key: bytes) -> list[ManifestMismatch]:
    """Verify a manifest.sig against the key.

    Checks:
    1. manifest_hmac integrity (detects tampering with the files dict).
    2. Each listed .safetensors file's HMAC against its current contents.

    Returns a list of ManifestMismatch (empty list means all checks pass).
    """
    mismatches: list[ManifestMismatch] = []

    manifest = load_manifest(manifest_path)
    files: dict[str, str] = manifest.get("files", {})
    stored_manifest_hmac: str = manifest.get("manifest_hmac", "")

    expected_manifest_hmac = _hmac_bytes(_serialize_files(files), key)
    if not hmac.compare_digest(stored_manifest_hmac, expected_manifest_hmac):
        mismatches.append(
            ManifestMismatch(
                path="manifest.sig",
                reason="manifest_hmac mismatch — manifest may have been tampered with",
            )
        )
        return mismatches

    checkpoint_dir = os.path.dirname(manifest_path)
    for filename, expected_hmac in files.items():
        full_path = os.path.join(checkpoint_dir, filename)
        if not os.path.exists(full_path):
            mismatches.append(
                ManifestMismatch(path=filename, reason=f"file not found at {full_path!r}")
            )
            continue
        actual_hmac = _hmac_file(full_path, key)
        if not hmac.compare_digest(actual_hmac, expected_hmac):
            mismatches.append(
                ManifestMismatch(
                    path=filename, reason="HMAC mismatch — file may have been tampered with"
                )
            )

    return mismatches
