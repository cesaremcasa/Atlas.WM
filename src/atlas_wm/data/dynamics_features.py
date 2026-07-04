"""Engineered per-step dynamics features for physics identification (v4 B10).

The v4 re-baseline (MODEL_CARD) showed a raw-sequence GRU scoring negative
R² on physics parameters that a closed-form estimator recovers at R² = 0.87
from the same data. The estimator's advantage is not capacity — it is the
FEATURES: per-step velocity-decay ratios with excitation and interaction
gating. This module hands those features to the learned belief encoder.

All quantities are computed in raw environment units (the ratios and
inverse-square terms are physics-meaningful there), from normalized windows.

Feature vector per step (N_FEATURES = 27), computed at positions 2..K−1 of a
K-frame window (two leading frames provide velocities and accelerations):

    0-1   agent velocity v_k
    2-5   box0 / box1 velocities
    6     friction decay ratio  <v_k, pre_k> / ||pre_k||²  (the oracle
          feature; pre_k = v_{k−1} + 0.8·u_k), clipped to [0, 1.5]
    7     ratio validity flag (excitation above threshold, off boundary,
          both boxes beyond the gravity gate — mirrors the oracle's filters)
    8     excitation magnitude ||pre_k||
    9-11  distances agent↔box0, agent↔box1, box0↔box1 (at step start)
    12-14 box acceleration projections onto the direction of the attracting
          body (gravity ∝ G/d² signals): box0→agent, box1→agent, box0→box1
    15-17 inverse-square distance regressors 1/d², clipped
    18-25 action one-hot u_k
    26    running MEDIAN of valid ratios up to step k — the oracle's
          sufficient statistic handed to the GRU (0 until first valid step)
"""

from __future__ import annotations

import torch

from atlas_wm.data.dataset import DEFAULT_OBS_SCALE

N_FEATURES = 27
ACTION_GAIN = 0.8
_FORCES = torch.tensor(
    [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)], dtype=torch.float32
)
_MIN_EXCITATION = 0.25  # ||pre||² below this makes the ratio ill-conditioned
_BOUNDARY = 0.5  # wall-clamp band; bounces there reflect velocity
_DIST_GATE = 5.0  # boxes nearer than this bias the ratio via gravity (oracle default)


def build_dynamics_features(
    obs_window: torch.Tensor,
    action_window: torch.Tensor,
    dt: float = 0.5,
    obs_scale: float = DEFAULT_OBS_SCALE,
    grid_size: float = 20.0,
) -> torch.Tensor:
    """[B, K, 6] normalized obs + [B, K, 8] one-hot actions → [B, K−2, N_FEATURES].

    Positions in the window are same-episode consecutive frames;
    ``action_window[:, k]`` is the action taken at frame k.
    """
    obs = obs_window * obs_scale  # raw units
    b, k, _ = obs.shape
    if k < 3:
        raise ValueError(f"dynamics features need K >= 3 frames, got {k}")

    pos = obs.reshape(b, k, 3, 2)  # agent, box0, box1
    vel = (pos[:, 1:] - pos[:, :-1]) / dt  # [B, K-1, 3, 2]; vel[t] moves frame t -> t+1

    forces = _FORCES.to(obs.device)
    # Alignment (dataset convention: action_window[:, j] is the action taken
    # AT frame j, producing frame j+1). Step s outputs vel[:, s+1] — the
    # transition frame s+1 -> s+2 — which is driven by the action taken at
    # frame s+1, i.e. action_window[:, s+1] == action_window[:, 1:-1].
    # (An earlier version used [:, 2:], pairing each ratio with the NEXT
    # step's action; every ratio was meaningless and friction_agent probed
    # negative while the action-free gravity features worked fine.)
    v_k = vel[:, 1:]  # [B, K-2, 3, 2]
    v_prev = vel[:, :-1]  # [B, K-2, 3, 2]
    aligned_actions = action_window[:, 1:-1]  # [B, K-2, 8]
    u = aligned_actions @ forces  # [B, K-2, 2]

    agent_v = v_k[:, :, 0]
    pre = v_prev[:, :, 0] + ACTION_GAIN * u
    pre_sq = (pre * pre).sum(-1)
    ratio_raw = (pre * agent_v).sum(-1) / pre_sq.clamp_min(1e-8)

    # Validity: enough excitation, and neither endpoint on the wall-clamp
    # boundary (observable bounces reflect velocity and poison the ratio).
    start_pos = pos[:, 1:-1, 0]  # agent at step start
    end_pos = pos[:, 2:, 0]
    on_boundary = (
        (start_pos <= _BOUNDARY + 1e-6)
        | (start_pos >= grid_size - _BOUNDARY - 1e-6)
        | (end_pos <= _BOUNDARY + 1e-6)
        | (end_pos >= grid_size - _BOUNDARY - 1e-6)
    ).any(dim=-1)
    # Distances at step start (frame k-1 relative to v_k's step).
    p_start = pos[:, 1:-1]  # [B, K-2, 3, 2]
    d_a0 = (p_start[:, :, 0] - p_start[:, :, 1]).norm(dim=-1)
    d_a1 = (p_start[:, :, 0] - p_start[:, :, 2]).norm(dim=-1)
    d_bb = (p_start[:, :, 1] - p_start[:, :, 2]).norm(dim=-1)

    # Validity mirrors the oracle's full filter set, including the gravity
    # gate: with a box inside _DIST_GATE the ratio is gravity-biased, and a
    # GRU aggregates by weighted mean — one biased "valid" step poisons it.
    valid = (pre_sq > _MIN_EXCITATION) & ~on_boundary & (torch.minimum(d_a0, d_a1) > _DIST_GATE)
    ratio = torch.where(valid, ratio_raw.clamp(0.0, 1.5), torch.zeros_like(ratio_raw))

    # Running median of valid ratios — the oracle's sufficient statistic.
    # K is small (<= ~20), so a per-step loop is cheap and stays vectorized
    # over the batch.
    n_steps = ratio.shape[1]
    running_median = torch.zeros_like(ratio)
    big = torch.where(valid, ratio, torch.full_like(ratio, float("nan")))
    for t in range(n_steps):
        running_median[:, t] = big[:, : t + 1].nanmedian(dim=1).values
    running_median = torch.nan_to_num(running_median, nan=0.0)

    # Box acceleration projections onto the attracting direction — the
    # gravity signal (boxes are moved only by gravity + friction).
    acc = (v_k - v_prev) / dt  # [B, K-2, 3, 2]

    def _proj(acc_body: torch.Tensor, from_p: torch.Tensor, to_p: torch.Tensor) -> torch.Tensor:
        direction = to_p - from_p
        unit = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return (acc_body * unit).sum(-1)

    g_b0_a = _proj(acc[:, :, 1], p_start[:, :, 1], p_start[:, :, 0])
    g_b1_a = _proj(acc[:, :, 2], p_start[:, :, 2], p_start[:, :, 0])
    g_b0_b1 = _proj(acc[:, :, 1], p_start[:, :, 1], p_start[:, :, 2])

    inv_sq = lambda d: (1.0 / d.pow(2).clamp_min(1.0)).clamp_max(1.0)  # noqa: E731

    feats = torch.cat(
        [
            agent_v,  # 2
            v_k[:, :, 1],  # 2
            v_k[:, :, 2],  # 2
            ratio.unsqueeze(-1),  # 1
            valid.float().unsqueeze(-1),  # 1
            pre_sq.clamp_min(0).sqrt().unsqueeze(-1),  # 1
            torch.stack([d_a0, d_a1, d_bb], dim=-1),  # 3
            torch.stack([g_b0_a, g_b1_a, g_b0_b1], dim=-1),  # 3
            torch.stack([inv_sq(d_a0), inv_sq(d_a1), inv_sq(d_bb)], dim=-1),  # 3
            aligned_actions,  # 8
            running_median.unsqueeze(-1),  # 1
        ],
        dim=-1,
    )
    assert feats.shape[-1] == N_FEATURES
    return feats
