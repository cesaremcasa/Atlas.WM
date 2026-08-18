"""Security, reproducibility, and path contracts for CI workflows."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github/workflows"
FIXTURES = ROOT / "tests/fixtures/ci_contract"
PINNED_ACTIONS = {
    "actions/checkout": "11bd71901bbe5b1630ceea73d27597364c9af683",
    "actions/setup-python": "a26af69be951a213d495a4c3e4e4022e16d87065",
    "actions/upload-artifact": "ea165f8d65b6e75b540449e92b4886f43607fa02",
    "astral-sh/setup-uv": "bd01e18f51369d5a26f1651c3cb451d3417e3bba",
}
SHA_RE = re.compile(r"[0-9a-f]{40}\Z")
CLEAN_GATE = 'test -z "$(git status --porcelain --untracked-files=all)"'


def _workflow_files() -> list[Path]:
    return sorted({*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")})


def _on_config(workflow: dict[str, Any]) -> dict[str, Any]:
    # PyYAML 5.x parses the YAML 1.1 key ``on`` as boolean True.
    return workflow.get("on", workflow.get(True, {}))


def _uses(workflow: dict[str, Any]) -> Iterator[tuple[str, str, dict[str, Any]]]:
    for job_id, job in workflow.get("jobs", {}).items():
        if "uses" in job:
            yield f"job {job_id}", job["uses"], job
        for index, step in enumerate(job.get("steps", [])):
            if "uses" in step:
                yield f"job {job_id} step {index}", step["uses"], step


def _runs(workflow: dict[str, Any]) -> Iterator[str]:
    for job in workflow.get("jobs", {}).values():
        if "run" in job:
            yield job["run"]
        for step in job.get("steps", []):
            if "run" in step:
                yield step["run"]


def _walk_values(value: Any) -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key), child
            yield from _walk_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_values(child)


def _validate_uses(workflow: dict[str, Any], path: Path) -> None:
    for location, reference, owner in _uses(workflow):
        if not isinstance(reference, str) or "@" not in reference:
            raise ValueError(f"malformed action reference at {path}:{location}")
        action, sha = reference.rsplit("@", 1)
        if action not in PINNED_ACTIONS:
            raise ValueError(f"unapproved action at {path}:{location}: {action}")
        if sha.startswith(("v", "main", "master")):
            raise ValueError(f"unverified SHA at {path}:{location}: {sha}")
        if not SHA_RE.fullmatch(sha):
            raise ValueError(f"malformed SHA at {path}:{location}: {sha}")
        if sha != PINNED_ACTIONS[action]:
            raise ValueError(f"unverified SHA at {path}:{location}: {action}")
        if (
            action == "actions/checkout"
            and owner.get("with", {}).get("persist-credentials") is not False
        ):
            raise ValueError(f"checkout credentials must be disabled at {path}:{location}")


def _validate_common_structure(workflow: dict[str, Any], path: Path) -> None:
    if workflow.get("permissions", {}) != {"contents": "read"}:
        raise ValueError(f"workflow permissions are not read-only: {path}")
    for key, value in _walk_values(workflow):
        if key == "continue-on-error":
            raise ValueError(f"continue-on-error is forbidden: {path}")
        if isinstance(value, str) and ("|| true" in value or "set +e" in value):
            raise ValueError(f"masked failure is forbidden: {path}")
    _validate_uses(workflow, path)


def _validate_runtime_structure(
    workflow: dict[str, Any], path: Path, *, require_schedule: bool
) -> None:
    on_config = _on_config(workflow)
    if require_schedule and ("schedule" not in on_config or not on_config["schedule"]):
        raise ValueError(f"scheduled trigger missing: {path}")
    if require_schedule and "workflow_dispatch" not in on_config:
        raise ValueError(f"manual trigger missing: {path}")
    if require_schedule and on_config["schedule"] != [{"cron": "0 8 * * MON"}]:
        raise ValueError(f"unexpected canary schedule: {path}")

    for job in workflow["jobs"].values():
        steps = job.get("steps", [])
        setup_python = [
            step for step in steps if step.get("uses", "").startswith("actions/setup-python@")
        ]
        setup_uv = [
            step for step in steps if step.get("uses", "").startswith("astral-sh/setup-uv@")
        ]
        if not setup_python or setup_python[0].get("with", {}).get("python-version") != "3.11.15":
            raise ValueError(f"Python pin missing or mutable: {path}")
        if not setup_uv:
            raise ValueError(f"setup-uv missing: {path}")
        uv_with = setup_uv[0].get("with", {})
        if uv_with.get("version") != "0.10.10" or uv_with.get("enable-cache") is not True:
            raise ValueError(f"uv version/cache pin missing: {path}")
        if uv_with.get("cache-dependency-glob") != "uv.lock":
            raise ValueError(f"uv cache key must be uv.lock: {path}")
        if not any("uv sync --locked --extra dev" in run for run in _runs({"jobs": {"job": job}})):
            raise ValueError(f"locked uv sync missing: {path}")

    text = path.read_text()
    if "pip install" in text:
        raise ValueError(f"unlocked pip install found: {path}")


def _validate_canary_structure(workflow: dict[str, Any], path: Path) -> None:
    for job in workflow["jobs"].values():
        steps = job.get("steps", [])
        if not steps or steps[-1].get("run") != CLEAN_GATE:
            raise ValueError(f"mandatory terminal clean gate missing: {path}")
        if path.name == "chaos-physics.yml":
            if not any("uv run python scripts/chaos_physics.py" in run for run in _runs(workflow)):
                raise ValueError(f"chaos path missing: {path}")
        if path.name == "train-canary.yml":
            if not any(
                "uv run pytest tests/test_determinism_canary.py tests/test_training_canary.py"
                in run
                for run in _runs(workflow)
            ):
                raise ValueError(f"training canary paths missing: {path}")


def validate_workflow(path: Path, *, require_runtime: bool = True) -> None:
    workflow = yaml.safe_load(path.read_text())
    _validate_common_structure(workflow, path)
    if require_runtime:
        _validate_runtime_structure(
            workflow,
            path,
            require_schedule=path.name in {"chaos-physics.yml", "train-canary.yml"},
        )
        if path.name in {"chaos-physics.yml", "train-canary.yml"}:
            _validate_canary_structure(workflow, path)


def test_current_workflows_satisfy_ci_contract():
    """Every checked-in YAML workflow satisfies the full structural contract."""
    for path in _workflow_files():
        validate_workflow(path)


def test_ci_and_canaries_use_locked_uv():
    """All install paths use the committed uv lock, never editable pip installs."""
    for name in ("ci.yml", "chaos-physics.yml", "train-canary.yml"):
        text = (WORKFLOWS / name).read_text()
        assert "uv sync --locked --extra dev" in text
        assert "pip install" not in text
        assert 'python-version: "3.11.15"' in text


@pytest.mark.parametrize(
    ("fixture", "message"),
    (
        ("mutable_action.yaml", "unverified SHA"),
        ("job_reusable_main.yml", "unapproved action"),
        ("unapproved_action.yml", "unapproved action"),
        ("malformed_sha.yml", "malformed SHA"),
        ("continue_on_error.yml", "continue-on-error"),
        ("masked_failure.yml", "masked failure"),
    ),
)
def test_negative_workflow_fixtures_are_rejected(fixture, message):
    """PoCs for mutable, unapproved, malformed, and masked workflow bypasses."""
    with pytest.raises(ValueError, match=message):
        validate_workflow(FIXTURES / fixture, require_runtime=False)
