"""Release staging, package-content, and clean-install checks."""

from __future__ import annotations

import gzip
import io
import json
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
from build_release import (
    EXPECTED_PYTHON,
    PASSTHROUGH_ENV,
    SDIST_NAME,
    VERSION,
    WHEEL_NAME,
    _archive_members,
    build_artifacts,
    validate_staging,
)

ROOT = Path(__file__).resolve().parents[1]


def _isolated_env(root: Path) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key in PASSTHROUGH_ENV}
    env.setdefault("PATH", os.defpath)
    root.mkdir(parents=True, exist_ok=True)
    home = root / "home"
    cache = root / "cache"
    config = root / "config"
    tmp = root / "tmp"
    home.mkdir()
    cache.mkdir()
    config.mkdir()
    tmp.mkdir()
    env.update(
        {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "TMPDIR": str(tmp),
            "TMP": str(tmp),
            "TEMP": str(tmp),
            "XDG_CACHE_HOME": str(cache),
            "XDG_CONFIG_HOME": str(config),
            "UV_CACHE_DIR": str(cache / "uv"),
            "PIP_CACHE_DIR": str(cache / "pip"),
            "PIP_CONFIG_FILE": os.devnull,
            "UV_NO_CONFIG": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": "0",
            "SOURCE_DATE_EPOCH": "0",
            "TZ": "UTC",
        }
    )
    return env


def _clean_install(staging: Path, artifact_name: str, env_dir: Path) -> None:
    python = env_dir / "bin/python"
    env = _isolated_env(env_dir.parent / f"{env_dir.name}-subprocess")
    if sys.version_info[:3] != EXPECTED_PYTHON:
        raise AssertionError(f"tests require Python {EXPECTED_PYTHON}, got {sys.version_info[:3]}")
    subprocess.run(["uv", "venv", "--python", sys.executable, str(env_dir)], check=True, env=env)
    subprocess.run(
        ["uv", "pip", "sync", "--python", str(python), str(ROOT / "requirements.lock")],
        check=True,
        env=env,
    )
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            "--no-build-isolation",
            str(staging / artifact_name),
        ],
        check=True,
        env=env,
    )
    smoke = """
import os
from pathlib import Path
import atlas_wm
from atlas_wm.checkpointing.io import load_checkpoint
from atlas_wm.environments.cruel_gridworld import CruelGridworld
assert atlas_wm.__version__ == "4.0.1"
obs, _ = CruelGridworld().reset(seed=42)
assert obs.shape == (6,)
checkpoint = Path(os.environ["ATLAS_ROOT"]) / "artifacts/article-audit-2026-07-21/gridworld/checkpoints/active-engineered-w40-seed42.safetensors"
state, metadata = load_checkpoint(checkpoint.as_posix(), expected_model_class="PhysicsBeliefEncoder", strict_env=False, allow_unsigned=True)
assert state and metadata["atlas_schema_version"] == "3.0.0"
"""
    subprocess.run(
        [str(python), "-c", smoke],
        check=True,
        env={**env, "ATLAS_ROOT": str(ROOT)},
    )


def test_release_staging_is_reproducible_and_exact(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_hashes = build_artifacts(ROOT, first)
    second_hashes = build_artifacts(ROOT, second)

    assert first_hashes == second_hashes
    assert {path.name for path in first.iterdir()} == {
        WHEEL_NAME,
        SDIST_NAME,
        "sbom.json",
        "SHA256SUMS",
    }
    validate_staging(ROOT, first)
    assert (
        json.loads((first / "sbom.json").read_text())["metadata"]["component"]["version"] == VERSION
    )
    with zipfile.ZipFile(first / WHEEL_NAME) as wheel:
        metadata = wheel.read("atlas_wm-4.0.1.dist-info/METADATA").decode()
    assert "Version: 4.0.1" in metadata


def _write_zip(path: Path, entries: list[tuple[str, bytes, int | None]]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload, mode in entries:
            info = zipfile.ZipInfo(name)
            if mode is not None:
                info.external_attr = mode << 16
            archive.writestr(info, payload)


def _write_tar(path: Path, members: list[tarfile.TarInfo], payloads: dict[str, bytes]) -> None:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        for member in members:
            archive.addfile(member, io.BytesIO(payloads[member.name]))
    with path.open("wb") as output:
        with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as compressed:
            compressed.write(raw.getvalue())


def test_archive_manifest_rejects_stale_member(tmp_path):
    archive = tmp_path / "stale.whl"
    _write_zip(archive, [("good.txt", b"ok", None), ("stale.txt", b"old", None)])
    with pytest.raises(ValueError, match="members differ"):
        _archive_members(archive, {"good.txt"})


def test_archive_rejects_traversal_and_mode_drift(tmp_path):
    traversal = tmp_path / "traversal.whl"
    _write_zip(traversal, [("../escape.txt", b"no", None)])
    with pytest.raises(ValueError, match="unsafe archive member"):
        _archive_members(traversal)

    mode_drift = tmp_path / "mode.whl"
    _write_zip(mode_drift, [("good.txt", b"no", 0o100755)])
    with pytest.raises(ValueError, match="mode"):
        _archive_members(mode_drift)


def test_archive_rejects_symlink_and_embedded_host_secret(tmp_path):
    symlink = tmp_path / "symlink.tar.gz"
    link = tarfile.TarInfo("link")
    link.type = tarfile.SYMTYPE
    link.linkname = "target"
    with pytest.raises(ValueError, match="symlink/hardlink/special"):
        _write_tar(symlink, [link], {"link": b""})
        _archive_members(symlink)

    secret = tmp_path / "secret.whl"
    _write_zip(secret, [("metadata.txt", b"token in /Users/host/project", None)])
    with pytest.raises(ValueError, match="forbidden host/secret"):
        _archive_members(secret)


@pytest.mark.parametrize(
    "credential",
    (
        b"github_pat_1234567890abcdefghij",
        b"ghp_1234567890abcdefghij",
        b"gho_1234567890abcdefghij",
        b"ghu_1234567890abcdefghij",
        b"ghs_1234567890abcdefghij",
        b"ghr_1234567890abcdefghij",
        b"glpat-1234567890abcdefghij",
        b"npm_1234567890abcdefghij",
        b"xoxb-1234567890",
        b"sk-1234567890abcdefghij",
        b"xai-1234567890abcdefghij",
        b"AKIA1234567890ABCDEF",
        b"-----BEGIN PRIVATE KEY-----",
    ),
)
def test_archive_rejects_high_signal_credentials(tmp_path, credential):
    archive = tmp_path / "credential.whl"
    _write_zip(archive, [("metadata.txt", credential, None)])
    with pytest.raises(ValueError, match="forbidden host/secret|high-signal secret"):
        _archive_members(archive)


def test_untracked_source_cannot_enter_release_build(tmp_path):
    stale = ROOT / "src/atlas_wm/_release_stale_probe.py"
    stale.write_text("# must not be packaged\n")
    try:
        with pytest.raises(ValueError, match="clean Git tree"):
            build_artifacts(ROOT, tmp_path / "staging")
    finally:
        stale.unlink()


def test_wheel_and_sdist_install_in_clean_locked_envs(tmp_path):
    staging = tmp_path / "staging"
    build_artifacts(ROOT, staging)
    _clean_install(staging, WHEEL_NAME, tmp_path / "wheel-env")
    _clean_install(staging, SDIST_NAME, tmp_path / "sdist-env")
