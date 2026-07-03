"""Deterministic seeding for the training pipelines (v4 B3, AD-7).

The determinism canary tests only covered eval-mode forward passes; training
itself was entirely unseeded (model init, batch order), so no checkpoint was
reproducible. Both trainers now call :func:`set_seed` before building models
and pass the returned generator (plus :func:`seed_worker`) to their
DataLoaders.

Scope: bit-identical reproducibility is guaranteed on CPU (where CI runs the
canary). CUDA kernels may be nondeterministic; for GPU-exact runs also enable
``torch.use_deterministic_algorithms(True)`` and cuDNN deterministic mode.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> torch.Generator:
    """Seed python, numpy and torch; return a generator for DataLoader shuffling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: derive per-worker seeds from torch's seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
