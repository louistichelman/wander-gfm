"""Deterministic eval subsample helpers.

Used by Wander loaders and LP baselines so a CLI cap picks the same
query/node subset across epochs and models.
"""

from __future__ import annotations

import zlib
from typing import Optional

import torch
from torch import Tensor


def dataset_eval_subsample_seed(base_seed: int, dataset_name: str) -> int:
    """Stable per-dataset seed so eval subsamples are fixed across epochs/models."""
    return (base_seed + zlib.adler32(dataset_name.encode())) % (2**31)


def subsample_indices(
    indices: Tensor,
    max_samples: Optional[int],
    subsample_seed: Optional[int] = None,
) -> Tensor:
    """Return a subset of indices, capped at ``max_samples`` (or all if None).

    When ``subsample_seed`` is set, the same subset is chosen on every call
    (fixed across epoch evaluations). Otherwise subsampling is random each call.
    """
    n = len(indices)
    if n == 0:
        return indices
    k = n if max_samples is None else min(max_samples, n)
    if subsample_seed is not None:
        g = torch.Generator()
        g.manual_seed(subsample_seed)
        perm = torch.randperm(n, generator=g)[:k]
    else:
        perm = torch.randperm(n)[:k]
    return indices[perm]
