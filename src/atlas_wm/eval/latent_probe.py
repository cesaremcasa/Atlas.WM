"""Latent probing for variable-physics identifiability (Block 12).

Thesis under test (AD-2): when the environment's physics vary per episode,
the variable parameters (gravity, friction) should be linearly decodable from
``z_static_slow`` — the soft-constrained "slow" sub-space — and *not* from the
other latents. A high R² probing ``z_static_slow`` against the ground-truth
physics, paired with a low R² probing ``z_static_immutable``, is direct evidence
that the hybrid static decomposition routes variable physics where intended.

The probe is a closed-form ridge regression (no sklearn dependency). Features
are standardized for numerical stability; R² is reported per target on a
held-out split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ProbeResult:
    """Outcome of a single linear probe.

    Attributes:
        r2_per_target: R² for each target column on the held-out split.
        r2_mean: Mean R² across targets.
        latent_key: Which latent sub-space was probed.
        target_names: Names of the probed targets, aligned with r2_per_target.
    """

    r2_per_target: np.ndarray
    r2_mean: float
    latent_key: str
    target_names: list[str]

    def __str__(self) -> str:
        rows = "\n".join(
            f"    {name:18s} R²={r2:+.4f}"
            for name, r2 in zip(self.target_names, self.r2_per_target)
        )
        return f"Probe[{self.latent_key}] mean R²={self.r2_mean:+.4f}\n{rows}"


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Closed-form ridge regression with a bias term.

    Solves ``(XᵀX + αI) W = Xᵀy`` where X is the bias-augmented design matrix.
    The bias column is excluded from regularization.

    Args:
        x: [N, d_feat] design matrix.
        y: [N, d_target] targets.
        alpha: L2 regularization strength.

    Returns:
        [d_feat + 1, d_target] weight matrix; the last row is the bias.
    """
    n = x.shape[0]
    x_aug = np.concatenate([x, np.ones((n, 1))], axis=1)  # [N, d_feat+1]
    d = x_aug.shape[1]
    reg = alpha * np.eye(d)
    reg[-1, -1] = 0.0  # do not regularize the bias term
    gram = x_aug.T @ x_aug + reg
    w = np.linalg.solve(gram, x_aug.T @ y)
    return w


def predict_ridge(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Apply a ridge weight matrix from :func:`fit_ridge` to features ``x``."""
    n = x.shape[0]
    x_aug = np.concatenate([x, np.ones((n, 1))], axis=1)
    preds: np.ndarray = x_aug @ w
    return preds


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-target coefficient of determination R².

    Returns 0.0 for any target whose true variance is ~0 (R² is undefined there).
    """
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    out: np.ndarray = np.zeros_like(ss_tot)
    nonzero = ss_tot > 1e-12
    out[nonzero] = 1.0 - ss_res[nonzero] / ss_tot[nonzero]
    return out


@torch.no_grad()
def encode_latent(
    encoder: nn.Module,
    obs: np.ndarray,
    latent_key: str = "z_static_slow",
    batch_size: int = 4096,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Run ``encoder`` over ``obs`` and return the requested latent as a numpy array.

    Args:
        encoder: A ContinuousEncoder (or compatible) returning a dict of latents.
        obs: [N, input_dim] observations.
        latent_key: Key into the encoder's output dict.
        batch_size: Mini-batch size for encoding.
        device: Torch device.

    Returns:
        [N, d_latent] numpy array of the requested latent.
    """
    encoder.eval()
    encoder.to(device)
    chunks: list[np.ndarray] = []
    x = torch.as_tensor(obs, dtype=torch.float32)
    for start in range(0, len(x), batch_size):
        batch = x[start : start + batch_size].to(device)
        out = encoder(batch)[latent_key]
        chunks.append(out.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _split_indices(
    n: int,
    train_frac: float,
    episode_ids: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Train/test index split for probe fitting.

    Without ``episode_ids`` the split is sequential (legacy behavior). With
    them, whole episodes are assigned to one side (shuffled, seeded): probe
    rows from the same episode share a physics label — and belief windows
    overlap by K−1 steps — so a row-level split leaks the label across the
    boundary and inflates R² (v4 B5, roadmap finding M3).
    """
    if episode_ids is None:
        split = int(train_frac * n)
        if split < 1 or split >= n:
            raise ValueError(f"train_frac={train_frac} yields an empty split for N={n}")
        idx = np.arange(n)
        return idx[:split], idx[split:]

    episode_ids = np.asarray(episode_ids)
    if len(episode_ids) != n:
        raise ValueError(f"episode_ids length {len(episode_ids)} != N={n}")
    unique_eps = np.unique(episode_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_eps)
    n_train_eps = int(train_frac * len(unique_eps))
    if n_train_eps < 1 or n_train_eps >= len(unique_eps):
        raise ValueError(
            f"train_frac={train_frac} yields an empty episode split for {len(unique_eps)} episodes"
        )
    train_eps = set(unique_eps[:n_train_eps].tolist())
    mask = np.array([e in train_eps for e in episode_ids])
    idx = np.arange(n)
    return idx[mask], idx[~mask]


def probe_latent(
    encoder: nn.Module,
    obs: np.ndarray,
    targets: np.ndarray,
    latent_key: str = "z_static_slow",
    target_names: list[str] | None = None,
    alpha: float = 1.0,
    train_frac: float = 0.8,
    device: torch.device | str = "cpu",
    episode_ids: np.ndarray | None = None,
) -> ProbeResult:
    """Fit a linear probe from a latent sub-space to ground-truth targets.

    Features are standardized using train-split statistics. The probe is fit on
    the first ``train_frac`` of the data and scored on the remainder.

    Args:
        encoder: Encoder producing a dict of latents.
        obs: [N, input_dim] observations.
        targets: [N, d_target] ground-truth values (e.g. physics parameters).
        latent_key: Latent sub-space to probe.
        target_names: Optional names, length d_target, for reporting.
        alpha: Ridge regularization strength.
        train_frac: Fraction of data used to fit the probe.
        device: Torch device for encoding.

    Returns:
        ProbeResult with per-target and mean R² on the held-out split.
    """
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim == 1:
        targets = targets[:, None]
    if target_names is None:
        target_names = [f"target_{i}" for i in range(targets.shape[1])]
    if len(target_names) != targets.shape[1]:
        raise ValueError(f"target_names length {len(target_names)} != n_targets {targets.shape[1]}")

    feats = encode_latent(encoder, obs, latent_key=latent_key, device=device).astype(np.float64)

    tr_idx, te_idx = _split_indices(len(feats), train_frac, episode_ids)
    x_tr, x_te = feats[tr_idx], feats[te_idx]
    y_tr, y_te = targets[tr_idx], targets[te_idx]

    # Standardize features with train-split statistics.
    mu = x_tr.mean(axis=0)
    sigma = x_tr.std(axis=0) + 1e-8
    x_tr = (x_tr - mu) / sigma
    x_te = (x_te - mu) / sigma

    w = fit_ridge(x_tr, y_tr, alpha=alpha)
    y_pred = predict_ridge(x_te, w)
    r2 = r2_score(y_te, y_pred)

    return ProbeResult(
        r2_per_target=r2,
        r2_mean=float(r2.mean()),
        latent_key=latent_key,
        target_names=list(target_names),
    )


def probe_from_arrays(
    feats: np.ndarray,
    targets: np.ndarray,
    latent_key: str = "z_static_slow",
    target_names: list[str] | None = None,
    alpha: float = 1.0,
    train_frac: float = 0.8,
    episode_ids: np.ndarray | None = None,
) -> ProbeResult:
    """Fit a linear probe directly from pre-computed feature arrays.

    Like :func:`probe_latent` but skips the encoding step — useful when the
    latent has already been computed (e.g. PhysicsBeliefEncoder outputs).

    Args:
        feats: [N, d_latent] pre-computed latent features.
        targets: [N, d_target] ground-truth values.
        latent_key: Label for reporting only.
        target_names: Optional names for each target column.
        alpha: Ridge regularization strength.
        train_frac: Fraction used to fit the probe.

    Returns:
        ProbeResult with per-target and mean R² on the held-out split.
    """
    feats = np.asarray(feats, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim == 1:
        targets = targets[:, None]
    if target_names is None:
        target_names = [f"target_{i}" for i in range(targets.shape[1])]

    tr_idx, te_idx = _split_indices(len(feats), train_frac, episode_ids)
    x_tr, x_te = feats[tr_idx], feats[te_idx]
    y_tr, y_te = targets[tr_idx], targets[te_idx]

    mu = x_tr.mean(axis=0)
    sigma = x_tr.std(axis=0) + 1e-8
    x_tr = (x_tr - mu) / sigma
    x_te = (x_te - mu) / sigma

    w = fit_ridge(x_tr, y_tr, alpha=alpha)
    y_pred = predict_ridge(x_te, w)
    r2 = r2_score(y_te, y_pred)

    return ProbeResult(
        r2_per_target=r2,
        r2_mean=float(r2.mean()),
        latent_key=latent_key,
        target_names=list(target_names),
    )
