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
ALLOWED_JOB_PERMISSIONS = ({}, {"contents": "read"})
INSTALL_COMMAND = "uv sync --locked --extra dev"
INSTALL_RE = re.compile(
    r"\b(?:uv\s+(?:sync|add|pip\s+install)|pip\s+install|"
    r"python\s+-m\s+pip\s+install)\b"
)


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
    if "defaults" in workflow or "working-directory" in workflow:
        raise ValueError(f"workflow execution defaults are forbidden: {path}")
    for job_id, job in workflow.get("jobs", {}).items():
        if "defaults" in job or "working-directory" in job:
            raise ValueError(f"job execution defaults are forbidden: {path}:{job_id}")
        permissions = job.get("permissions")
        if permissions is not None and permissions not in ALLOWED_JOB_PERMISSIONS:
            raise ValueError(f"job permissions are elevated or not allowlisted: {path}:{job_id}")
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
        if len(setup_python) != 1:
            raise ValueError(f"exactly one setup-python is required: {path}")
        if len(setup_uv) != 1:
            raise ValueError(f"exactly one setup-uv is required: {path}")
        if setup_python[0].get("with", {}).get("python-version") != "3.11.15":
            raise ValueError(f"Python pin missing or mutable: {path}")
        uv_with = setup_uv[0].get("with", {})
        if uv_with.get("version") != "0.10.10" or uv_with.get("enable-cache") is not True:
            raise ValueError(f"uv version/cache pin missing: {path}")
        if uv_with.get("cache-dependency-glob") != "uv.lock":
            raise ValueError(f"uv cache key must be uv.lock: {path}")
        _validate_installation(job, path)


def _validate_installation(job: dict[str, Any], path: Path) -> None:
    installation_steps: list[dict[str, Any]] = []
    for step in job.get("steps", []):
        run = step.get("run")
        if not isinstance(run, str) or not INSTALL_RE.search(run):
            continue
        installation_steps.append(step)
        if run.strip() != INSTALL_COMMAND or "\n" in run or "\r" in run:
            raise ValueError(f"installation command must be exactly {INSTALL_COMMAND!r}: {path}")
        if any(key in step for key in ("if", "continue-on-error", "shell")):
            raise ValueError(f"installation step must be unconditional/default shell: {path}")

    if len(installation_steps) != 1:
        raise ValueError(f"exactly one installation step is required: {path}")


def _validate_canary_structure(workflow: dict[str, Any], path: Path) -> None:
    for job in workflow["jobs"].values():
        steps = job.get("steps", [])
        if not steps or steps[-1].get("run") != CLEAN_GATE:
            raise ValueError(f"mandatory terminal clean gate missing: {path}")
        if "if" in steps[-1] or "continue-on-error" in steps[-1]:
            raise ValueError(f"clean gate must be unconditional: {path}")
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


def validate_workflow(
    path: Path, *, require_runtime: bool = True, require_clean_gate: bool = False
) -> None:
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
    elif require_clean_gate:
        _validate_canary_structure(workflow, path)
    else:
        # Negative fixtures that pass the common checks still exercise the
        # setup-count and pin checks without needing a complete schedule.
        _validate_runtime_structure(workflow, path, require_schedule=False)


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
    ("fixture", "message", "clean_gate"),
    (
        ("mutable_action.yaml", "unverified SHA", False),
        ("job_reusable_main.yml", "unapproved action", False),
        ("unapproved_action.yml", "unapproved action", False),
        ("malformed_sha.yml", "malformed SHA", False),
        ("continue_on_error.yml", "continue-on-error", False),
        ("masked_failure.yml", "masked failure", False),
        ("job_permissions_write.yml", "job permissions", False),
        ("second_setup_python.yml", "exactly one setup-python", False),
        ("second_setup_uv.yml", "exactly one setup-uv", False),
        ("clean_if_false.yml", "unconditional", True),
        ("install_masked_fallback.yml", "installation command must be exactly", False),
        ("install_multiline_fallback.yml", "installation command must be exactly", False),
        ("second_unlocked_sync.yml", "installation command must be exactly", False),
        ("uv_pip_install.yml", "installation command must be exactly", False),
        ("custom_shell_install.yml", "installation step must be unconditional", False),
        ("workflow_defaults_shell.yml", "workflow execution defaults", False),
        ("job_defaults_shell.yml", "job execution defaults", False),
        ("workflow_defaults_workdir.yml", "workflow execution defaults", False),
        ("job_defaults_workdir.yml", "job execution defaults", False),
    ),
)
def test_negative_workflow_fixtures_are_rejected(fixture, message, clean_gate):
    """PoCs for mutable, unapproved, malformed, and masked workflow bypasses."""
    with pytest.raises(ValueError, match=message):
        validate_workflow(
            FIXTURES / fixture,
            require_runtime=False,
            require_clean_gate=clean_gate,
        )
