"""Helpers for extra real-world NC train queries (hide + supervise)."""

from __future__ import annotations

import torch
from torch import Tensor


def nc_train_query_count(n_train: int, batch_size: int, frac: float, cap: int) -> int:
    """How many train nodes to hide / supervise on a real-world NC forward.

    ``max(min(int(n_train * frac), cap), batch_size)``, clamped to ``n_train``.
    Disabled (returns ``min(batch_size, n_train)``) when ``frac <= 0`` or ``cap <= 0``.
    """
    if n_train <= 0:
        return 0
    batch_size = max(0, int(batch_size))
    if float(frac) <= 0.0 or int(cap) <= 0:
        return min(batch_size, n_train)
    k = min(int(n_train * float(frac)), int(cap))
    k = max(k, min(batch_size, n_train))
    return min(k, n_train)


@torch.no_grad()
def sample_nc_train_query_ids(train_idx: Tensor, batch_indices: Tensor, k: int) -> Tensor:
    """Sample ``k`` train ids that always include ``batch_indices``.

    Extra ids are drawn uniformly from ``train_idx`` minus the batch. All batch
    nodes are kept even if that exceeds ``k`` (they are the walk seeds).
    """
    batch = torch.unique(batch_indices)
    if k <= int(batch.numel()):
        return batch
    device = train_idx.device
    batch = batch.to(device=device, dtype=train_idx.dtype)
    extra_pool = train_idx[~torch.isin(train_idx, batch)]
    n_extra = int(k) - int(batch.numel())
    if extra_pool.numel() == 0 or n_extra <= 0:
        return batch
    if extra_pool.numel() <= n_extra:
        extras = extra_pool
    else:
        perm = torch.randperm(extra_pool.numel(), device=device)[:n_extra]
        extras = extra_pool[perm]
    return torch.cat([batch, extras], dim=0)
