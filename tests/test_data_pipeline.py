"""Data-pipeline contract tests (v4 B2).

Covers the three failure modes from the roadmap:
- C5: normalization must happen in memory, never by mutating split files;
  directories normalized in place by pre-v4 train.py must be rejected.
- M4: re-running generate_data.py must trigger a re-split instead of
  silently training on stale processed data.
- H5 lives in tests/test_physics_belief.py (episode windowing).
"""

import numpy as np
import pytest
from split_data import raw_fingerprint, split_data  # noqa: E402

from atlas_wm.data.dataset import ATLASDataset, reject_legacy_normalized
from atlas_wm.data.episode_dataset import EpisodeATLASDataset


def _write_split(tmp_path, split="train", n=32, low=0.0, high=20.0, seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.uniform(low, high, size=(n, 6)).astype(np.float32)
    np.save(tmp_path / f"{split}_obs.npy", obs)
    np.save(tmp_path / f"{split}_actions.npy", rng.standard_normal((n, 8)).astype(np.float32))
    np.save(tmp_path / f"{split}_next_obs.npy", obs + 0.1)
    return obs


def _write_raw(raw_dir, n=50, n_episodes=5, seed=0):
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    np.save(raw_dir / "observations.npy", rng.uniform(0, 20, size=(n, 6)))
    np.save(raw_dir / "actions.npy", rng.standard_normal((n, 8)))
    np.save(raw_dir / "next_observations.npy", rng.uniform(0, 20, size=(n, 6)))
    np.save(raw_dir / "episode_ids.npy", np.repeat(np.arange(n_episodes), n // n_episodes))


class TestInMemoryNormalization:
    def test_dataset_scales_obs_to_unit_range(self, tmp_path):
        raw = _write_split(tmp_path)
        ds = ATLASDataset(str(tmp_path), split="train")
        item = ds[0]
        assert item["obs"].max() <= 1.0 + 1e-6
        np.testing.assert_allclose(ds.observations, raw / 20.0, rtol=1e-6)

    def test_files_on_disk_are_never_modified(self, tmp_path):
        raw = _write_split(tmp_path)
        ATLASDataset(str(tmp_path), split="train")
        ATLASDataset(str(tmp_path), split="train")  # loading twice must not compound
        on_disk = np.load(tmp_path / "train_obs.npy")
        np.testing.assert_array_equal(on_disk, raw)

    def test_episode_dataset_applies_same_scale(self, tmp_path):
        raw = _write_split(tmp_path)
        np.save(tmp_path / "train_episode_ids.npy", np.zeros(len(raw), dtype=np.int64))
        eds = EpisodeATLASDataset(str(tmp_path), split="train", window_k=3)
        ds = ATLASDataset(str(tmp_path), split="train")
        np.testing.assert_allclose(eds.obs, ds.observations, rtol=1e-6)

    def test_legacy_normalized_dir_is_rejected(self, tmp_path):
        _write_split(tmp_path)
        (tmp_path / ".normalized").touch()
        with pytest.raises(RuntimeError, match="normalized in place"):
            ATLASDataset(str(tmp_path), split="train")
        with pytest.raises(RuntimeError, match="normalized in place"):
            reject_legacy_normalized(str(tmp_path))


class TestSplitFingerprint:
    def test_skip_when_raw_unchanged(self, tmp_path, capsys):
        raw, processed = tmp_path / "raw", tmp_path / "processed"
        _write_raw(raw)
        split_data(raw_dir=str(raw), processed_dir=str(processed))
        first = np.load(processed / "train_obs.npy")
        split_data(raw_dir=str(raw), processed_dir=str(processed))
        assert "unchanged — skipping" in capsys.readouterr().out
        np.testing.assert_array_equal(np.load(processed / "train_obs.npy"), first)

    def test_resplit_when_raw_changes(self, tmp_path):
        raw, processed = tmp_path / "raw", tmp_path / "processed"
        _write_raw(raw, seed=0)
        split_data(raw_dir=str(raw), processed_dir=str(processed))
        stale = np.load(processed / "train_obs.npy")
        _write_raw(raw, seed=1)  # generate_data.py always overwrites raw
        split_data(raw_dir=str(raw), processed_dir=str(processed))
        fresh = np.load(processed / "train_obs.npy")
        assert not np.array_equal(stale, fresh), "re-split did not pick up new raw data"

    def test_fingerprint_distinguishes_absent_files(self, tmp_path):
        raw = tmp_path / "raw"
        _write_raw(raw)
        with_ids = raw_fingerprint(str(raw))
        (raw / "episode_ids.npy").unlink()
        assert raw_fingerprint(str(raw)) != with_ids

    def test_resplit_clears_legacy_normalized_sentinel(self, tmp_path):
        raw, processed = tmp_path / "raw", tmp_path / "processed"
        _write_raw(raw)
        processed.mkdir()
        (processed / ".normalized").touch()
        split_data(raw_dir=str(raw), processed_dir=str(processed))
        assert not (processed / ".normalized").exists()
