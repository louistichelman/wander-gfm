"""Original-graph visit counts from Wander walk tensors (no prune)."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from torch import Tensor


def original_graph_visit_stats_np(walks_np: np.ndarray, num_nodes: int) -> Dict[str, float]:
    """CPU unique-node stats from a walk array ``[T, bsize, n_walks, L]``."""
    n_full = int(num_nodes)
    if walks_np.size == 0 or walks_np.shape[0] == 0:
        n_1 = 0
        n_tot = 0
    else:
        n_1 = int(np.unique(walks_np[0]).size)
        n_tot = int(np.unique(walks_np).size)
    scale = 100.0 / n_full if n_full > 0 else 0.0
    return {
        "node_n_graph_1step": float(n_1),
        "node_n_graph_total": float(n_tot),
        "node_pct_graph_1step": n_1 * scale,
        "node_pct_graph_total": n_tot * scale,
    }


def original_graph_visit_stats(walks: Tensor, num_nodes: int) -> Dict[str, float]:
    """Unique original-graph node ids in ``walks`` (global ids, before prune).

    ``walks`` is ``[T, bsize, n_walks, L]``. ``node_pct_graph_*`` divides by
    the full graph ``num_nodes``, unlike the pruned-subgraph ``node_pct_*``.
    """
    n_full = int(num_nodes)
    if walks.numel() == 0 or walks.shape[0] == 0:
        n_1 = 0
        n_tot = 0
    else:
        n_1 = int(walks[0].unique().numel())
        n_tot = int(walks.unique().numel())
    scale = 100.0 / n_full if n_full > 0 else 0.0
    return {
        "node_n_graph_1step": float(n_1),
        "node_n_graph_total": float(n_tot),
        "node_pct_graph_1step": n_1 * scale,
        "node_pct_graph_total": n_tot * scale,
    }


def size_tier_walk_params(num_nodes: int, walk_num_div: int = 1) -> tuple[int, int]:
    """Match ``eval_one_dataset.sh`` SIZE_TIER_WALKS (length stays 32 then 128)."""
    if num_nodes > 40000:
        n, length = 256, 128
    elif num_nodes > 10000:
        n, length = 128, 128
    elif num_nodes > 1000:
        n, length = 64, 128
    else:
        n, length = 32, 32
    div = max(1, int(walk_num_div))
    return max(1, n // div), length


def unique_nodes(walks_np: np.ndarray, refinement: Optional[int] = None) -> np.ndarray:
    """Unique global node ids in a walk tensor ``[T, G, S, L]``."""
    sl = walks_np if refinement is None else walks_np[refinement : refinement + 1]
    if sl.size == 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(sl.ravel())


def n_in_mask(node_ids: np.ndarray, mask: np.ndarray) -> int:
    """Count unique ids in ``node_ids`` that are True in ``mask``."""
    if node_ids.size == 0 or mask.size == 0:
        return 0
    valid = np.unique(node_ids[(node_ids >= 0) & (node_ids < mask.size)])
    if valid.size == 0:
        return 0
    return int(mask[valid].sum())
