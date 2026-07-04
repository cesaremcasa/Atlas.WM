"""Per-step dynamics features for MuJoCo physics identification (v4.1).

The measured iteration ledger (see v4.1 part-1 commit): speed-ratio features
fail (Coulomb friction decelerates linearly, not exponentially); the AGENT
is nearly frictionless (slide joints absorb gravity, normal force ~0); only
the BOXES feel real friction. Sufficient statistics:

- BOX coast deceleration while sliding free of the agent -> mu*g product
  (gravity alone is structurally unidentifiable here);
- box speed response per unit agent speed during contact -> mass;
- running medians of both, handed to the GRU (B10 pattern).

Feature vector per step (N_MJ_FEATURES = 16), positions 1..K-1 of a window:
    0-1   agent velocity        2   agent speed
    3     noop/coast flag (action index 8, if present)
    4-5   box speeds            6-8 distances a-b0, a-b1, b0-b1
    9     box coast decel (valid steps else 0)   10  its validity flag
    11    running median of valid box decels  (mu*g estimate so far)
    12    contact response bsp/sp (valid else 0) 13  its validity flag
    14    running median of valid contact responses (mass proxy so far)
    15    commanded-direction dot agent velocity
"""

from __future__ import annotations

import torch

N_MJ_FEATURES = 16
_DIRS = torch.tensor(
    [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)],
    dtype=torch.float32,
)
_DIRS = _DIRS / _DIRS.norm(dim=1, keepdim=True)


def _running_median(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(values)
    masked = torch.where(valid, values, torch.full_like(values, float("nan")))
    for t in range(values.shape[1]):
        out[:, t] = masked[:, : t + 1].nanmedian(dim=1).values
    return torch.nan_to_num(out, nan=0.0)


def build_mujoco_features(
    obs_window: torch.Tensor,
    action_window: torch.Tensor,
    dt: float = 0.1,
    obs_scale: float = 1.2,
) -> torch.Tensor:
    """[B, K, 6] normalized obs + [B, K, A] one-hot actions -> [B, K-2, 16]."""
    obs = obs_window * obs_scale
    b, k, _ = obs.shape
    if k < 3:
        raise ValueError(f"mujoco features need K >= 3 frames, got {k}")
    pos = obs.reshape(b, k, 3, 2)
    vel = (pos[:, 1:] - pos[:, :-1]) / dt  # [B, K-1, 3, 2]

    v_k = vel[:, 1:]  # step s -> transition s+1->s+2
    v_prev = vel[:, :-1]
    sp = v_k[:, :, 0].norm(dim=-1)
    acts = action_window[:, 1:-1]  # aligned: action at frame s+1 (B10 lesson)
    n_act = acts.shape[-1]
    noop = acts[:, :, 8] if n_act > 8 else torch.zeros_like(sp)
    u = acts[:, :, :8] @ _DIRS.to(obs.device)

    p_start = pos[:, 1:-1]
    d_a0 = (p_start[:, :, 0] - p_start[:, :, 1]).norm(dim=-1)
    d_a1 = (p_start[:, :, 0] - p_start[:, :, 2]).norm(dim=-1)
    d_bb = (p_start[:, :, 1] - p_start[:, :, 2]).norm(dim=-1)

    bsp0, bsp0_prev = v_k[:, :, 1].norm(dim=-1), v_prev[:, :, 1].norm(dim=-1)
    bsp1, bsp1_prev = v_k[:, :, 2].norm(dim=-1), v_prev[:, :, 2].norm(dim=-1)

    # Box coast decel: box sliding, agent far -> decel = mu*g (Coulomb).
    free0 = (d_a0 > 0.20) & (bsp0_prev > 0.08)
    free1 = (d_a1 > 0.20) & (bsp1_prev > 0.08)
    dec0 = (bsp0_prev - bsp0) / dt
    dec1 = (bsp1_prev - bsp1) / dt
    decel = torch.where(free0, dec0, torch.where(free1, dec1, torch.zeros_like(dec0)))
    decel_valid = free0 | free1
    decel = torch.where(decel_valid, decel.clamp(-2.0, 15.0), torch.zeros_like(decel))

    # Contact response: box speed generated per unit agent speed -> ~1/mass.
    contact0 = (d_a0 < 0.16) & (sp > 0.05)
    contact1 = (d_a1 < 0.16) & (sp > 0.05)
    resp = torch.where(
        contact0,
        bsp0 / sp.clamp_min(1e-6),
        torch.where(contact1, bsp1 / sp.clamp_min(1e-6), torch.zeros_like(sp)),
    ).clamp(0.0, 5.0)
    resp_valid = contact0 | contact1

    feats = torch.cat(
        [
            v_k[:, :, 0],  # 2
            sp.unsqueeze(-1),  # 1
            noop.unsqueeze(-1),  # 1
            torch.stack([bsp0, bsp1], dim=-1),  # 2
            torch.stack([d_a0, d_a1, d_bb], dim=-1),  # 3
            decel.unsqueeze(-1),  # 1
            decel_valid.float().unsqueeze(-1),  # 1
            _running_median(decel, decel_valid).unsqueeze(-1),  # 1
            resp.unsqueeze(-1),  # 1
            resp_valid.float().unsqueeze(-1),  # 1
            _running_median(resp, resp_valid).unsqueeze(-1),  # 1
            (u * v_k[:, :, 0]).sum(-1, keepdim=True),  # 1
        ],
        dim=-1,
    )
    assert feats.shape[-1] == N_MJ_FEATURES
    return feats
