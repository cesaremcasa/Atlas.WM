"""Release metadata and checkpoint compatibility guards for v4.0.1."""

import tomllib
from pathlib import Path

import pytest

import atlas_wm
from atlas_wm.checkpointing.io import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
LEGACY_BELIEF_CHECKPOINT = (
    ROOT
    / "artifacts/article-audit-2026-07-21/gridworld/checkpoints/"
    / "active-engineered-w40-seed42.safetensors"
)


def test_package_version_matches_project_metadata():
    """The import-time version must remain aligned with the build metadata."""
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project_version = tomllib.load(handle)["project"]["version"]

    assert project_version == "4.0.1"
    assert atlas_wm.__version__ == project_version


def test_v4_legacy_belief_checkpoint_remains_loadable():
    """A pre-4.0.1 safetensors checkpoint still loads after the release bump.

    The artifact intentionally carries ``atlas_schema_version=3.0.0``: the
    package release version and serialized checkpoint schema are independent,
    so a patch release must not invalidate existing research artifacts.
    """
    assert LEGACY_BELIEF_CHECKPOINT.is_file()

    with pytest.warns(RuntimeWarning, match="without signature verification"):
        state_dict, metadata = load_checkpoint(
            str(LEGACY_BELIEF_CHECKPOINT),
            expected_model_class="PhysicsBeliefEncoder",
            strict_env=False,
            allow_unsigned=True,
        )

    assert state_dict
    assert metadata["model_class"] == "PhysicsBeliefEncoder"
    assert metadata["atlas_schema_version"] == "3.0.0"
    assert metadata["git_sha"] == "db10166"
