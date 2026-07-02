"""Smoke test for the friction_agent identifiability oracle (v4 B4).

Guards the retraction evidence: if a physics/env change silently makes
friction_agent unidentifiable again (or breaks the estimator), this trips.
The full run (400 episodes x 50 steps) scores R² = 0.85; this reduced run
uses a permissive threshold to stay fast and non-flaky.
"""

import sys

sys.path.insert(0, "scripts")
from oracle_friction_agent import run_oracle  # noqa: E402


def test_friction_agent_is_identifiable_from_random_policy():
    result = run_oracle(episodes=80, steps=50, seed=123)
    assert result["episodes_used"] >= 50, "estimator skipped too many episodes"
    assert result["r2"] > 0.5, (
        f"friction_agent oracle R² = {result['r2']:.3f} — identifiability regressed "
        "(full-scale reference: R² = 0.85 at 400 episodes)"
    )
    assert result["mae"] < 0.02
