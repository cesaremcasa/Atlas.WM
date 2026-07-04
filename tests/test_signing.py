"""Tests for HMAC-SHA256 checkpoint signing (Block 4 / AD-4)."""

import json
import os

import torch
from safetensors.torch import save_file

from atlas_wm.checkpointing.signing import (
    build_manifest,
    load_manifest,
    verify_manifest,
    write_manifest,
)

KEY = bytes.fromhex("deadbeef" * 8)


def _write_fake_checkpoint(dir_path: str, name: str = "model.safetensors") -> str:
    path = os.path.join(dir_path, name)
    save_file({"w": torch.randn(4, 4)}, path)
    return path


class TestBuildManifest:
    def test_includes_all_safetensors(self, tmp_path):
        _write_fake_checkpoint(str(tmp_path), "a.safetensors")
        _write_fake_checkpoint(str(tmp_path), "b.safetensors")
        manifest = build_manifest(str(tmp_path), KEY)
        assert set(manifest["files"].keys()) == {"a.safetensors", "b.safetensors"}

    def test_excludes_non_safetensors(self, tmp_path):
        _write_fake_checkpoint(str(tmp_path), "model.safetensors")
        (tmp_path / "readme.txt").write_text("hi")
        manifest = build_manifest(str(tmp_path), KEY)
        assert "readme.txt" not in manifest["files"]

    def test_manifest_hmac_present(self, tmp_path):
        _write_fake_checkpoint(str(tmp_path))
        manifest = build_manifest(str(tmp_path), KEY)
        assert "manifest_hmac" in manifest
        assert len(manifest["manifest_hmac"]) == 64

    def test_empty_dir_produces_empty_files(self, tmp_path):
        manifest = build_manifest(str(tmp_path), KEY)
        assert manifest["files"] == {}


class TestVerifyManifest:
    def test_clean_checkpoint_passes(self, tmp_path):
        _write_fake_checkpoint(str(tmp_path))
        manifest = build_manifest(str(tmp_path), KEY)
        mp = str(tmp_path / "manifest.sig")
        write_manifest(mp, manifest)
        mismatches = verify_manifest(mp, KEY)
        assert mismatches == []

    def test_tampered_file_detected(self, tmp_path):
        ckpt_path = _write_fake_checkpoint(str(tmp_path))
        manifest = build_manifest(str(tmp_path), KEY)
        mp = str(tmp_path / "manifest.sig")
        write_manifest(mp, manifest)

        with open(ckpt_path, "r+b") as f:
            f.seek(10)
            f.write(b"\xff\xff")

        mismatches = verify_manifest(mp, KEY)
        assert len(mismatches) == 1
        assert mismatches[0].path == "model.safetensors"
        assert "HMAC mismatch" in mismatches[0].reason

    def test_wrong_key_detected(self, tmp_path):
        _write_fake_checkpoint(str(tmp_path))
        manifest = build_manifest(str(tmp_path), KEY)
        mp = str(tmp_path / "manifest.sig")
        write_manifest(mp, manifest)
        wrong_key = bytes(32)
        mismatches = verify_manifest(mp, wrong_key)
        assert len(mismatches) >= 1
        assert mismatches[0].path == "manifest.sig"

    def test_missing_file_detected(self, tmp_path):
        ckpt_path = _write_fake_checkpoint(str(tmp_path))
        manifest = build_manifest(str(tmp_path), KEY)
        mp = str(tmp_path / "manifest.sig")
        write_manifest(mp, manifest)
        os.remove(ckpt_path)
        mismatches = verify_manifest(mp, KEY)
        assert len(mismatches) == 1
        assert "not found" in mismatches[0].reason

    def test_tampered_manifest_detected(self, tmp_path):
        _write_fake_checkpoint(str(tmp_path))
        manifest = build_manifest(str(tmp_path), KEY)
        mp = str(tmp_path / "manifest.sig")
        write_manifest(mp, manifest)

        raw = load_manifest(mp)
        filename = next(iter(raw["files"]))
        raw["files"][filename] = "a" * 64
        with open(mp, "w") as f:
            json.dump(raw, f)

        mismatches = verify_manifest(mp, KEY)
        assert len(mismatches) >= 1
        assert mismatches[0].path == "manifest.sig"


class TestManifestRoundtrip:
    def test_write_then_load_stable(self, tmp_path):
        _write_fake_checkpoint(str(tmp_path))
        manifest = build_manifest(str(tmp_path), KEY)
        mp = str(tmp_path / "manifest.sig")
        write_manifest(mp, manifest)
        loaded = load_manifest(mp)
        assert loaded["files"] == manifest["files"]
        assert loaded["manifest_hmac"] == manifest["manifest_hmac"]

    def test_deterministic_for_same_file(self, tmp_path):
        _write_fake_checkpoint(str(tmp_path))
        m1 = build_manifest(str(tmp_path), KEY)
        m2 = build_manifest(str(tmp_path), KEY)
        assert m1["files"] == m2["files"]
        assert m1["manifest_hmac"] == m2["manifest_hmac"]


class TestFailClosedLoading:
    """v4 B17 (finding H4): require_signature=True turns every fail-open
    path into an error."""

    def _save(self, tmp_path, name="model.safetensors"):
        import torch

        from atlas_wm.checkpointing.io import save_checkpoint

        path = str(tmp_path / name)
        save_checkpoint(
            {"w": torch.zeros(2)},
            path,
            {
                "model_class": "M",
                "git_sha": "x",
                "config_hash": "y",
                "env_hash": "z",
                "trained_at_utc": "t",
                "atlas_schema_version": "3.0.0",
            },
        )
        return path

    def test_missing_manifest_rejected(self, tmp_path):
        import pytest

        from atlas_wm.checkpointing.io import SignatureMismatch, load_checkpoint

        path = self._save(tmp_path)
        with pytest.raises(SignatureMismatch, match="no manifest"):
            load_checkpoint(
                path, expected_model_class="M", strict_env=False, require_signature=True
            )

    def test_missing_key_rejected(self, tmp_path, monkeypatch):
        import pytest

        from atlas_wm.checkpointing.io import SignatureMismatch, load_checkpoint
        from atlas_wm.checkpointing.signing import build_manifest, write_manifest

        path = self._save(tmp_path)
        monkeypatch.setenv("ATLAS_SIGNING_KEY", "aa" * 16)
        write_manifest(
            str(tmp_path / "manifest.sig"), build_manifest(str(tmp_path), bytes.fromhex("aa" * 16))
        )
        monkeypatch.delenv("ATLAS_SIGNING_KEY")
        with pytest.raises(SignatureMismatch, match="ATLAS_SIGNING_KEY"):
            load_checkpoint(
                path, expected_model_class="M", strict_env=False, require_signature=True
            )

    def test_unlisted_file_rejected(self, tmp_path, monkeypatch):
        import pytest

        from atlas_wm.checkpointing.io import SignatureMismatch, load_checkpoint
        from atlas_wm.checkpointing.signing import build_manifest, write_manifest

        self._save(tmp_path)  # signed below
        monkeypatch.setenv("ATLAS_SIGNING_KEY", "bb" * 16)
        write_manifest(
            str(tmp_path / "manifest.sig"), build_manifest(str(tmp_path), bytes.fromhex("bb" * 16))
        )
        evil = self._save(tmp_path, name="evil.safetensors")  # added AFTER signing
        with pytest.raises(SignatureMismatch, match="not listed"):
            load_checkpoint(
                evil, expected_model_class="M", strict_env=False, require_signature=True
            )

    def test_valid_signed_load_passes(self, tmp_path, monkeypatch):
        from atlas_wm.checkpointing.io import load_checkpoint
        from atlas_wm.checkpointing.signing import build_manifest, write_manifest

        path = self._save(tmp_path)
        monkeypatch.setenv("ATLAS_SIGNING_KEY", "cc" * 16)
        write_manifest(
            str(tmp_path / "manifest.sig"), build_manifest(str(tmp_path), bytes.fromhex("cc" * 16))
        )
        state, meta = load_checkpoint(
            path, expected_model_class="M", strict_env=False, require_signature=True
        )
        assert "w" in state and meta["model_class"] == "M"
