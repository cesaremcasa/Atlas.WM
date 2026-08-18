"""SBOM provenance and determinism guards."""

import json
from pathlib import Path

import pytest
from generate_sbom import build_sbom, validate_sbom, write_sbom

ROOT = Path(__file__).resolve().parents[1]


def test_committed_sbom_has_no_host_paths():
    """The committed SBOM must not contain environment-specific paths."""
    sbom = json.loads((ROOT / "sbom.json").read_text())
    validate_sbom(sbom)


@pytest.mark.parametrize(
    "host_path",
    (
        "file:///Users/example/project",
        "/Users/example/project",
        "/home/runner/work/project",
        "/workspace/project",
        "/workspaces/project",
    ),
)
def test_sbom_rejects_host_paths(host_path):
    """All known local/runner/workspace path forms must fail closed."""
    with pytest.raises(ValueError, match="forbidden host path"):
        validate_sbom({"metadata": {"host_path": host_path}})


def test_sbom_regeneration_is_byte_stable(tmp_path):
    """The same project and lock inputs produce identical bytes."""
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    write_sbom(ROOT / "pyproject.toml", ROOT / "requirements.lock", first_path)
    write_sbom(ROOT / "pyproject.toml", ROOT / "requirements.lock", second_path)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert json.loads(first_path.read_text()) == build_sbom(
        ROOT / "pyproject.toml", ROOT / "requirements.lock"
    )
