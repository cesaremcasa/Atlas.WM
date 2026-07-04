"""Immutable-anchor tests (v4 B9, finding C3).

The AD-2 passthrough guarantees z_static_immutable is constant across TIME,
but says nothing about content: the encoder's trivially optimal solution is
a constant vector — bit-identical, encoding nothing. The anchor enforces
content: invariant within an episode, variant across episodes. This is the
variance-floor test the red-team review found missing.
"""

import argparse

import numpy as np
import torch
import yaml
from evaluate import load_world_model
from train import train

from atlas_wm.data.dataset import stack_window
from atlas_wm.data.episode_dataset import EpisodeATLASDataset


def _make_identifiable_setup(tmp_path, n_eps=12, ep_len=24):
    """Synthetic data where episode identity is trivially observable:
    each episode's observations live around a distinct constant offset."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for split, eps in (("train", n_eps), ("val", 4)):
        rows = eps * ep_len
        ids = np.repeat(np.arange(eps), ep_len)
        offsets = rng.uniform(2, 18, size=(eps, 6)).astype(np.float32)
        obs = offsets[ids] + rng.uniform(-0.5, 0.5, size=(rows, 6)).astype(np.float32)
        np.save(data_dir / f"{split}_obs.npy", obs.clip(0, 20))
        np.save(
            data_dir / f"{split}_actions.npy",
            np.eye(8, dtype=np.float32)[rng.integers(0, 8, size=rows)],
        )
        np.save(data_dir / f"{split}_next_obs.npy", (obs + 0.05).clip(0, 20))
        np.save(data_dir / f"{split}_episode_ids.npy", ids)

    cfg = {
        "model": {
            "d_static_immutable": 8,
            "d_static_slow": 8,
            "d_dynamic": 32,
            "d_controllable": 16,
            "input_dim": 6,
            "frame_stack": 2,
        },
        "environment": {"action_space_size": 8},
        "data": {"data_dir": str(data_dir)},
        "training": {
            "seed": 42,
            "objective": "vicreg",
            "rollout_k": 2,
            "batch_size": 32,
            "learning_rate": 1.0e-3,
            "num_epochs": 12,
            "early_stopping_patience": 50,
            "grad_clip_norm": 0.5,
            "lr_scheduler_patience": 10,
            "lr_scheduler_factor": 0.5,
            "lambda_recon": 1.0,
            "lambda_slow_drift": 0.1,
            "lambda_imm_invariance": 1.0,
            "lambda_imm_variance": 1.0,
        },
        "checkpointing": {"checkpoint_dir": str(tmp_path / "ckpt")},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    return config_path, str(data_dir)


@torch.no_grad()
def imm_variance_ratio(encoder, data_dir, split="train", frame_stack=2, window_k=8):
    """Across-episode variance of per-episode-mean z_imm over mean
    within-episode variance. Collapsed-constant z_imm → ratio ≈ 0/0 → tiny
    numerator; informative z_imm → ratio ≫ 1."""
    ds = EpisodeATLASDataset(data_dir, split=split, window_k=window_k)
    per_episode: dict[int, list[np.ndarray]] = {}
    for i in range(len(ds)):
        item = ds[i]
        obs_w = item["obs_window"].unsqueeze(0)
        inputs = stack_window(obs_w, frame_stack)
        z_imm = encoder(inputs[:, 0])["z_static_immutable"].squeeze(0).numpy()
        ep = int(ds.episode_ids[int(ds.valid_indices[i])])
        per_episode.setdefault(ep, []).append(z_imm)

    means = np.stack([np.mean(v, axis=0) for v in per_episode.values()])
    within = float(np.mean([np.var(np.stack(v), axis=0).mean() for v in per_episode.values()]))
    across = float(means.var(axis=0).mean())
    return across / (within + 1e-12)


class TestImmutableAnchorVarianceFloor:
    def test_z_imm_distinguishes_episodes(self, tmp_path):
        config, data_dir = _make_identifiable_setup(tmp_path)
        ckpt = str(tmp_path / "model.safetensors")
        train(
            argparse.Namespace(
                config=str(config),
                max_steps=None,
                no_checkpoint=False,
                output_checkpoint=ckpt,
                seed=None,
            )
        )
        encoder, _, _, meta = load_world_model(ckpt)
        ratio = imm_variance_ratio(encoder, data_dir, frame_stack=int(meta["frame_stack"]))
        # Variance floor (C3): a collapsed-constant z_imm scores ~1 or below
        # (no structure beyond noise); an informative one separates episodes
        # by orders of magnitude on this trivially identifiable data.
        assert ratio > 10.0, (
            f"z_imm across/within episode variance ratio = {ratio:.2f} — "
            "the immutable latent is not encoding episode identity (C3 regression)"
        )
