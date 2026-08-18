"""Security and path contracts for the CI and scheduled canary workflows."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github/workflows"
PINNED_ACTIONS = {
    "actions/checkout": "11bd71901bbe5b1630ceea73d27597364c9af683",
    "actions/setup-python": "a26af69be951a213d495a4c3e4e4022e16d87065",
    "actions/upload-artifact": "ea165f8d65b6e75b540449e92b4886f43607fa02",
    "astral-sh/setup-uv": "bd01e18f51369d5a26f1651c3cb451d3417e3bba",
}


def _steps(workflow: dict) -> list[dict]:
    return [step for job in workflow["jobs"].values() for step in job.get("steps", [])]


def test_workflows_use_pinned_read_only_actions():
    """Every workflow action is pinned and checkout cannot persist credentials."""
    for path in sorted(WORKFLOWS.glob("*.yml")):
        workflow = yaml.safe_load(path.read_text())
        assert workflow["permissions"] == {"contents": "read"}, path
        for step in _steps(workflow):
            if "uses" not in step:
                continue
            action, sha = step["uses"].split("@", 1)
            assert action in PINNED_ACTIONS, f"unapproved action in {path}: {action}"
            assert sha == PINNED_ACTIONS[action], f"unverified SHA in {path}: {action}"
            assert re.fullmatch(r"[0-9a-f]{40}", sha)
            if action == "actions/checkout":
                assert step.get("with", {}).get("persist-credentials") is False


def test_ci_and_canaries_use_locked_uv():
    """All install paths use the committed uv lock, never editable pip installs."""
    for name in ("ci.yml", "chaos-physics.yml", "train-canary.yml"):
        text = (WORKFLOWS / name).read_text()
        assert "uv sync --locked --extra dev" in text
        assert "pip install" not in text
        assert 'python-version: "3.11.15"' in text


def test_canaries_run_real_paths_and_leave_checkout_clean():
    chaos = (WORKFLOWS / "chaos-physics.yml").read_text()
    train = (WORKFLOWS / "train-canary.yml").read_text()

    assert "uv run python scripts/chaos_physics.py" in chaos
    assert "uv run pytest tests/test_determinism_canary.py tests/test_training_canary.py" in train
    for text in (chaos, train):
        assert "git status --porcelain --untracked-files=all" in text
