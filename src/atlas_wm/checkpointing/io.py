"""Checkpoint I/O using safetensors only (AD-4).

All production checkpoint reads and writes go through this module.
Pickle-based serialization (pt/pth files) is explicitly forbidden in production paths.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

from safetensors.torch import load_file, save_file

REQUIRED_METADATA_KEYS = frozenset(
    {
        "model_class",
        "git_sha",
        "config_hash",
        "env_hash",
        "trained_at_utc",
        "atlas_schema_version",
    }
)


class UnsafeFormatError(Exception):
    """Raised when attempting to load a non-safetensors checkpoint (e.g. .pt).

    See scripts/migrate_pt_to_safetensors.py to convert legacy checkpoints.
    """


class SignatureMismatch(Exception):
    """Raised when checkpoint signature verification fails."""


class EnvHashMismatch(Exception):
    """Raised when checkpoint env_hash does not match the current environment."""


class MetadataError(ValueError):
    """Raised when required metadata keys are missing."""


def compute_config_hash(config: dict[str, Any]) -> str:
    """Return a stable SHA-256 hex digest of a config dict (JSON-canonical)."""
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def save_checkpoint(
    state_dict: dict,
    path: str,
    metadata: dict[str, str],
) -> None:
    """Save a model state_dict as a safetensors file with embedded metadata.

    Args:
        state_dict: Model state dict (str -> Tensor).
        path: Output path; must end with .safetensors.
        metadata: Dict[str, str] with at minimum REQUIRED_METADATA_KEYS.

    Raises:
        MetadataError: If any required key is missing from metadata.
        ValueError: If path does not end with .safetensors.
    """
    if not path.endswith(".safetensors"):
        raise ValueError(
            f"Checkpoint path must end with .safetensors, got: {path!r}. "
            "See scripts/migrate_pt_to_safetensors.py for legacy .pt migration."
        )

    missing = REQUIRED_METADATA_KEYS - set(metadata.keys())
    if missing:
        raise MetadataError(
            f"Missing required metadata keys: {sorted(missing)}. "
            f"Required: {sorted(REQUIRED_METADATA_KEYS)}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    save_file(state_dict, path, metadata=metadata)


def load_checkpoint(
    path: str,
    *,
    expected_model_class: str,
    strict_env: bool = True,
    current_env_hash: str | None = None,
    allow_unsigned: bool = False,
) -> tuple[dict, dict[str, str]]:
    """Load a safetensors checkpoint and validate metadata.

    Args:
        path: Path to the .safetensors file.
        expected_model_class: The model class name this checkpoint should contain.
        strict_env: If True, validate env_hash against current_env_hash.
        current_env_hash: Hash of the current environment (from env_hash.py).
        allow_unsigned: If True, skip signature verification (emits RuntimeWarning).

    Returns:
        (state_dict, metadata) tuple.

    Raises:
        UnsafeFormatError: If path ends with .pt or is not .safetensors.
        FileNotFoundError: If path does not exist.
        MetadataError: If model_class does not match expected_model_class.
        EnvHashMismatch: If strict_env=True and env hashes differ.
        SignatureMismatch: If signature verification fails (when enabled).
    """
    if path.endswith(".pt") or path.endswith(".pth"):
        raise UnsafeFormatError(
            f"Refusing to load {path!r}: .pt/.pth files use pickle and are unsafe. "
            "Run scripts/migrate_pt_to_safetensors.py to convert to safetensors. "
            "See docs/SECURITY.md for details."
        )

    if not path.endswith(".safetensors"):
        raise UnsafeFormatError(
            f"Refusing to load {path!r}: only .safetensors files are supported (AD-4)."
        )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path!r}")

    if allow_unsigned:
        import warnings

        warnings.warn(
            f"Loading {path!r} without signature verification. "
            "This is insecure in production. Set allow_unsigned=False.",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        _verify_signature_if_manifest_exists(path)

    state_dict = load_file(path)

    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as f:
        metadata = dict(f.metadata() or {})

    actual_class = metadata.get("model_class", "")
    if actual_class != expected_model_class:
        raise MetadataError(
            f"model_class mismatch: checkpoint has {actual_class!r}, "
            f"expected {expected_model_class!r}."
        )

    if strict_env and current_env_hash is not None:
        checkpoint_env = metadata.get("env_hash", "")
        if checkpoint_env and checkpoint_env != current_env_hash:
            raise EnvHashMismatch(
                f"env_hash mismatch: checkpoint has {checkpoint_env!r}, "
                f"current env is {current_env_hash!r}. "
                "Pass --allow-env-mismatch to override (emits warning)."
            )

    return state_dict, metadata


def _verify_signature_if_manifest_exists(path: str) -> None:
    """Verify checkpoint signature against manifest if manifest exists.

    This is a lightweight check; full signing is implemented in Block 4
    (signing.py). For now, if no manifest exists, we skip verification.
    """
    manifest_path = os.path.join(os.path.dirname(path), "manifest.sig")
    if not os.path.exists(manifest_path):
        return

    signing_key = os.environ.get("ATLAS_SIGNING_KEY")
    if not signing_key:
        import warnings

        warnings.warn(
            "ATLAS_SIGNING_KEY not set; skipping signature verification.",
            RuntimeWarning,
            stacklevel=3,
        )
        return

    try:
        from atlas_wm.checkpointing.signing import verify_manifest

        mismatches = verify_manifest(manifest_path, bytes.fromhex(signing_key))
        filename = os.path.basename(path)
        for m in mismatches:
            if m.path == filename:
                raise SignatureMismatch(f"Signature mismatch for {path!r}: {m.reason}")
    except ImportError:
        pass


def make_metadata(
    model_class: str,
    config: dict[str, Any],
    env_hash: str | None = None,
    git_sha: str | None = None,
) -> dict[str, str]:
    """Convenience helper to build a valid metadata dict."""
    if env_hash is None:
        from atlas_wm.checkpointing.env_hash import compute_env_hash

        env_hash = compute_env_hash()

    if git_sha is None:
        try:
            import subprocess

            git_sha = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
        except Exception:
            git_sha = "unknown"

    return {
        "model_class": model_class,
        "git_sha": git_sha,
        "config_hash": compute_config_hash(config),
        "env_hash": env_hash,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "atlas_schema_version": "3.0.0",
    }
