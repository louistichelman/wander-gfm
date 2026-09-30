"""Disconnected-grids label-propagation dataset (no features).

Five independent 25x25 4-neighbor grids, one class per grid. Features are
omitted by default. Every grid has a handful of train nodes; the rest of
the component is val/test. Solving the task requires carrying the train label
across the grid (diameter 48).

Cached under ``raw_data/GRIDS/processed/data.pt``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from . import motif_er

NAME = "Grids"
HAS_NODE_FEATURES = False
HAS_NODE_LABELS = True
HAS_PREDEFINED_NODE_SPLIT = True
HAS_PREDEFINED_EDGE_SPLIT = False

REGISTRY_KEY = "GRIDS"

GRID_H = 25
GRID_W = 25
N_GRIDS = 5
NUM_CLASSES = 5
NODES_PER_GRID = GRID_H * GRID_W
N_NODES = N_GRIDS * NODES_PER_GRID
DIAMETER = (GRID_H - 1) + (GRID_W - 1)

TRAIN_PER_GRID = 2
N_TRAIN = N_GRIDS * TRAIN_PER_GRID
N_VAL = 400
N_TEST = N_NODES - N_TRAIN - N_VAL

SEED = 0

CLASS_NAMES = tuple(f"c{i}" for i in range(NUM_CLASSES))


def _one_grid_edges(h: int, w: int) -> torch.Tensor:
    """Undirected 4-neighbor grid edges as a directed-pair list."""
    idx = torch.arange(h * w).view(h, w)
    horiz = torch.stack([idx[:, :-1].reshape(-1), idx[:, 1:].reshape(-1)], dim=0)
    vert = torch.stack([idx[:-1, :].reshape(-1), idx[1:, :].reshape(-1)], dim=0)
    return torch.cat([horiz, vert], dim=1)


def disjoint_grid_edge_index(n_grids: int, h: int, w: int) -> torch.Tensor:
    """Concatenate ``n_grids`` disconnected ``h x w`` lattices."""
    n_per = h * w
    pieces = [_one_grid_edges(h, w) + g * n_per for g in range(n_grids)]
    edge_index = torch.cat(pieces, dim=1)
    return to_undirected(edge_index, num_nodes=n_grids * n_per)


def build_grids_data(
    *,
    seed: int = SEED,
    h: int = GRID_H,
    w: int = GRID_W,
    n_grids: int = N_GRIDS,
    num_classes: int = NUM_CLASSES,
    train_per_grid: int = TRAIN_PER_GRID,
    n_val: int = N_VAL,
) -> Data:
    """Build the label-only disconnected-grid graph (no disk I/O)."""
    if n_grids != num_classes:
        raise ValueError(
            f"expected one grid per class, got n_grids={n_grids}, num_classes={num_classes}"
        )
    n_per = h * w
    n = n_grids * n_per
    n_train = n_grids * train_per_grid
    if n_train + n_val >= n:
        raise ValueError(f"n_train={n_train} + n_val={n_val} leaves no test nodes (n={n})")
    if train_per_grid < 1 or train_per_grid >= n_per:
        raise ValueError(
            f"train_per_grid must be in [1, {n_per - 1}], got {train_per_grid}"
        )

    g = torch.Generator().manual_seed(seed)
    edge_index = disjoint_grid_edge_index(n_grids, h, w)

    grid_classes = torch.arange(n_grids, dtype=torch.long)
    grid_classes = grid_classes[torch.randperm(n_grids, generator=g)]

    y = torch.empty(n, dtype=torch.long)
    grid_id = torch.empty(n, dtype=torch.long)
    for gid in range(n_grids):
        lo, hi = gid * n_per, (gid + 1) * n_per
        y[lo:hi] = int(grid_classes[gid])
        grid_id[lo:hi] = gid

    train_mask = torch.zeros(n, dtype=torch.bool)
    for gid in range(n_grids):
        lo, hi = gid * n_per, (gid + 1) * n_per
        pool = torch.arange(lo, hi)
        pick = pool[torch.randperm(int(pool.numel()), generator=g)[:train_per_grid]]
        train_mask[pick] = True

    remaining = (~train_mask).nonzero(as_tuple=False).view(-1)
    rest_perm = remaining[torch.randperm(int(remaining.numel()), generator=g)]
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    val_mask[rest_perm[:n_val]] = True
    test_mask[rest_perm[n_val:]] = True

    return Data(
        x=None,
        y=y,
        edge_index=edge_index,
        num_nodes=n,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        grid_id=grid_id,
        num_classes=num_classes,
        grid_h=h,
        grid_w=w,
        n_grids=n_grids,
    )


def load_base_dataset(data_dir: str, seed: int = SEED, *, force: bool = False) -> Data:
    del seed  # fixed SEED for reproducibility (cache key).
    path = motif_er.cache_path(data_dir, REGISTRY_KEY)

    def _build() -> Data:
        return build_grids_data(seed=SEED)

    return motif_er.build_or_load(path, _build, force=force)


def _class_counts(y: torch.Tensor) -> dict[str, int]:
    y = y.view(-1).long()
    names = CLASS_NAMES
    n_cls = int(y.max().item()) + 1 if y.numel() else 0
    if n_cls != len(names):
        names = tuple(f"c{i}" for i in range(n_cls))
    return {names[c]: int((y == c).sum().item()) for c in range(n_cls)}


def _hop_to_train(data: Data) -> torch.Tensor:
    """Manhattan distance from each node to the nearest train node in its grid."""
    w = int(data.grid_w)
    n_per = int(data.grid_h) * w
    n = int(data.num_nodes)
    local = torch.arange(n) % n_per
    r, c = local // w, local % w
    dist = torch.full((n,), n_per, dtype=torch.long)
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    grid_id = data.grid_id.view(-1).long()
    for t in train_idx.tolist():
        same = grid_id == int(grid_id[t])
        tr, tc = divmod(int(t) % n_per, w)
        cand = (r - tr).abs() + (c - tc).abs()
        dist = torch.where(same, torch.minimum(dist, cand), dist)
    return dist


def dataset_meta(data_dir: str) -> dict:
    """Lightweight metadata for the generate script (loads or builds cache)."""
    data = load_base_dataset(data_dir, force=False)
    y = data.y.view(-1).long()
    n = int(data.num_nodes)
    deg = data.edge_index[0].bincount(minlength=n).float()
    grid_id = data.grid_id.view(-1).long()
    n_grids = int(data.n_grids)
    train_grid = torch.zeros(n_grids, dtype=torch.bool)
    for gid in range(n_grids):
        train_grid[gid] = bool((data.train_mask & (grid_id == gid)).any())
    hops = _hop_to_train(data)
    y_tr = y[data.train_mask]
    train_per_class = {
        CLASS_NAMES[c]: int((y_tr == c).sum().item()) for c in range(NUM_CLASSES)
    }
    return {
        "registry_key": REGISTRY_KEY,
        "num_classes": NUM_CLASSES,
        "grid_h": int(data.grid_h),
        "grid_w": int(data.grid_w),
        "n_grids": n_grids,
        "diameter": DIAMETER,
        "class_counts": _class_counts(y),
        "n_edges": motif_er.num_undirected_edges(data.edge_index),
        "mean_degree": float(deg.mean().item()),
        "cache": str(Path(data_dir) / REGISTRY_KEY / "processed" / "data.pt"),
        "n": float(n),
        "n_train": float(int(data.train_mask.sum().item())),
        "n_val": int(data.val_mask.sum().item()),
        "n_test": int(data.test_mask.sum().item()),
        "split": f"{N_TRAIN}/{N_VAL}/{N_TEST}",
        "n_grids_with_train": int(train_grid.sum().item()),
        "train_per_class": train_per_class,
        "train_per_grid": TRAIN_PER_GRID,
        "max_hops_to_train": int(hops.max().item()),
        "mean_hops_to_train": float(hops.float().mean().item()),
        "has_node_features": HAS_NODE_FEATURES,
    }
