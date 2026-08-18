"""Release staging, package-content, and clean-install checks."""

from __future__ import annotations

import json
import os
import subprocess
import zipfile
from pathlib import Path

from build_release import SDIST_NAME, VERSION, WHEEL_NAME, build_artifacts, validate_staging

ROOT = Path(__file__).resolve().parents[1]


def _clean_install(staging: Path, artifact_name: str, env_dir: Path) -> None:
    python = env_dir / "bin/python"
    subprocess.run(["uv", "venv", "--python", "3.11", str(env_dir)], check=True)
    subprocess.run(
        ["uv", "pip", "sync", "--python", str(python), str(ROOT / "requirements.lock")],
        check=True,
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
        [str(python), "-c", smoke], check=True, env={**dict(os.environ), "ATLAS_ROOT": str(ROOT)}
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
    validate_staging(first)
    assert (
        json.loads((first / "sbom.json").read_text())["metadata"]["component"]["version"] == VERSION
    )
    with zipfile.ZipFile(first / WHEEL_NAME) as wheel:
        metadata = wheel.read("atlas_wm-4.0.1.dist-info/METADATA").decode()
    assert "Version: 4.0.1" in metadata


def test_wheel_and_sdist_install_in_clean_locked_envs(tmp_path):
    staging = tmp_path / "staging"
    build_artifacts(ROOT, staging)
    _clean_install(staging, WHEEL_NAME, tmp_path / "wheel-env")
    _clean_install(staging, SDIST_NAME, tmp_path / "sdist-env")
