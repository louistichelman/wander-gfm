"""Shared helpers for motif-on-ER expressivity datasets.

Generates an undirected Erdos-Renyi graph, labels nodes by participation in a
chosen motif (triangle or 4-cycle), attaches non-informative random features,
and a random 10/20/70 train/val/test split (no train rebalancing).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Set, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


FEAT_DIM = 16
DEFAULT_N_NODES = 5000
# Fixed eval graphs use a small labeled context (dense walk coverage on train).
TRAIN_SIZE = 50
VAL_FRAC = 0.20
# p tuned so motif-positive nodes are roughly 15-30% (see generate script stats).
TRIANGLE_ER_P = 0.003
C4_ER_P = 0.0015


def sample_er_undirected(
    n: int,
    p: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample an undirected simple ER graph; return ``edge_index`` [2, E]."""
    # Upper-triangle Bernoulli; vectorized via random scores.
    # For n=5000 this is ~12.5M candidate edges - fine in memory.
    iu = torch.triu_indices(n, n, offset=1)
    n_cand = iu.shape[1]
    keep = torch.rand(n_cand, generator=generator) < p
    src = iu[0, keep]
    dst = iu[1, keep]
    edge_index = torch.stack([src, dst], dim=0)
    return to_undirected(edge_index, num_nodes=n)


def _adjacency_sets(edge_index: torch.Tensor, n: int) -> list[Set[int]]:
    nbrs: list[Set[int]] = [set() for _ in range(n)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for u, v in zip(src, dst):
        if u == v:
            continue
        nbrs[u].add(v)
        nbrs[v].add(u)
    return nbrs


def nodes_in_triangles(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    """Boolean mask: node participates in at least one triangle."""
    nbrs = _adjacency_sets(edge_index, n)
    in_motif = torch.zeros(n, dtype=torch.bool)
    # Iterate each undirected edge once (u < v).
    for u in range(n):
        for v in nbrs[u]:
            if u >= v:
                continue
            common = nbrs[u] & nbrs[v]
            if not common:
                continue
            in_motif[u] = True
            in_motif[v] = True
            for w in common:
                in_motif[w] = True
    return in_motif


def nodes_in_c4(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    """Boolean mask: node participates in at least one 4-cycle.

    For every wedge a-u-b, if a and b share at least two common neighbors
    (including u), those common neighbors with a,b form C4s.
    """
    nbrs = _adjacency_sets(edge_index, n)
    in_motif = torch.zeros(n, dtype=torch.bool)
    for u in range(n):
        nbr_list = sorted(nbrs[u])
        for i, a in enumerate(nbr_list):
            for b in nbr_list[i + 1 :]:
                common = nbrs[a] & nbrs[b]
                if len(common) < 2:
                    continue
                in_motif[u] = True
                in_motif[a] = True
                in_motif[b] = True
                for c in common:
                    in_motif[c] = True
    return in_motif


def nodes_in_c5(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    """Boolean mask: node participates in at least one 5-cycle.

    Enumerate simple walks of length 4 and mark a C5 when the endpoints close.
    Cheap for small degree (planted graphs use d=3).
    """
    nbrs = _adjacency_sets(edge_index, n)
    in_motif = torch.zeros(n, dtype=torch.bool)
    for a in range(n):
        for b in nbrs[a]:
            if b == a:
                continue
            for c in nbrs[b]:
                if c == a or c == b:
                    continue
                for d in nbrs[c]:
                    if d == a or d == b or d == c:
                        continue
                    for e in nbrs[d]:
                        if e == a or e == b or e == c or e == d:
                            continue
                        if a not in nbrs[e]:
                            continue
                        in_motif[a] = True
                        in_motif[b] = True
                        in_motif[c] = True
                        in_motif[d] = True
                        in_motif[e] = True
    return in_motif


def make_random_masks(
    n: int,
    generator: torch.Generator,
    n_train: int = TRAIN_SIZE,
    val_frac: float = VAL_FRAC,
    n_val: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random partition: fixed ``n_train`` train nodes, then val and test.

    When ``n_val`` is set, use that exact val count and assign the remainder to
    test. Otherwise val size is ``round(n * val_frac)`` (legacy default).
    """
    if n_train < 1 or n_train >= n:
        raise ValueError(f"n_train must be in [1, n-1], got n_train={n_train}, n={n}")
    perm = torch.randperm(n, generator=generator)
    if n_val is not None:
        if n_val < 1:
            raise ValueError(f"n_val must be >= 1, got {n_val}")
        n_test = n - n_train - n_val
    else:
        n_val = int(round(n * val_frac))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        n_test = n - n_train - n_val
    if n_test < 1:
        raise ValueError(
            f"not enough nodes for train/val/test: n={n}, n_train={n_train}, n_val={n_val}"
        )

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train : n_train + n_val]] = True
    test_mask[perm[n_train + n_val :]] = True
    return train_mask, val_mask, test_mask


def label_stats(y: torch.Tensor, train_mask: torch.Tensor) -> Dict[str, float]:
    """Global and train-split positive rates for binary labels."""
    y = y.long().view(-1)
    n = int(y.numel())
    n_pos = int((y == 1).sum().item())
    n_neg = n - n_pos
    y_tr = y[train_mask]
    n_tr = int(y_tr.numel())
    n_pos_tr = int((y_tr == 1).sum().item()) if n_tr else 0
    n_neg_tr = n_tr - n_pos_tr
    return {
        "n": float(n),
        "n_pos": float(n_pos),
        "n_neg": float(n_neg),
        "pos_rate": float(n_pos / n) if n else 0.0,
        "n_train": float(n_tr),
        "n_pos_train": float(n_pos_tr),
        "n_neg_train": float(n_neg_tr),
        "pos_rate_train": float(n_pos_tr / n_tr) if n_tr else 0.0,
    }


def build_motif_data(
    *,
    n: int,
    p: float,
    seed: int,
    motif_fn: Callable[[torch.Tensor, int], torch.Tensor],
    feat_dim: int = FEAT_DIM,
) -> Data:
    g = torch.Generator().manual_seed(seed)
    edge_index = sample_er_undirected(n, p, g)
    in_motif = motif_fn(edge_index, n)
    y = in_motif.long()
    x = torch.randn(n, feat_dim, generator=g)
    train_mask, val_mask, test_mask = make_random_masks(n, g)
    return Data(
        x=x,
        y=y,
        edge_index=edge_index,
        num_nodes=n,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


def cache_path(data_dir: str, registry_key: str) -> Path:
    return Path(data_dir) / registry_key / "processed" / "data.pt"


def build_or_load(
    path: Path,
    build_fn: Callable[[], Data],
    *,
    force: bool = False,
) -> Data:
    if path.is_file() and not force:
        return torch.load(path, map_location="cpu", weights_only=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = build_fn()
    torch.save(data, path)
    if not path.is_file():
        raise RuntimeError(f"Failed to write motif cache: {path}")
    return data


def num_undirected_edges(edge_index: torch.Tensor) -> int:
    """Count undirected edges (assumes bidirectional storage)."""
    return int(edge_index.size(1) // 2)
