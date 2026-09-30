"""Cycles dataset: multi-class node labels for planted C3 / C4 / C5 blocks.

Disjoint planted motifs on a shared girth>=5 3-regular backbone.
Labels: 0=backbone, 1=triangle, 2=C4, 3=C5 (~600 nodes per motif class).
Node features: none by default (structure-only).
Node split: 500 train / 1000 val / 3500 test.
Cached under ``raw_data/CYCLES/processed/data.pt``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Data

from . import motif_er, motif_planted

NAME = "Cycles"
HAS_NODE_FEATURES = False
HAS_NODE_LABELS = True
HAS_PREDEFINED_NODE_SPLIT = True
HAS_PREDEFINED_EDGE_SPLIT = False

REGISTRY_KEY = "CYCLES"
N_NODES = motif_er.DEFAULT_N_NODES
DEGREE = motif_planted.DEFAULT_DEGREE
NODES_PER_CLASS = 600
NUM_CLASSES = 4
M3, M4, M5 = motif_planted.n_motifs_mixed_balanced(N_NODES, nodes_per_class=NODES_PER_CLASS)
SEED = 0
N_TRAIN = 500
N_VAL = 1000
N_TEST = N_NODES - N_TRAIN - N_VAL

CLASS_NAMES = ("backbone", "triangle", "c4", "c5")
FEAT_DIM = 0


def load_base_dataset(data_dir: str, seed: int = SEED, *, force: bool = False) -> Data:
    del seed  # fixed SEED for reproducibility (cache key).
    path = motif_er.cache_path(data_dir, REGISTRY_KEY)

    def _build() -> Data:
        return motif_planted.build_planted_mixed_data(
            n=N_NODES,
            m3=M3,
            m4=M4,
            m5=M5,
            seed=SEED,
            d=DEGREE,
            feat_dim=FEAT_DIM,
            n_train=N_TRAIN,
            n_val=N_VAL,
        )

    return motif_er.build_or_load(path, _build, force=force)


def _class_counts(y: torch.Tensor) -> dict[str, int]:
    y = y.view(-1).long()
    return {CLASS_NAMES[c]: int((y == c).sum().item()) for c in range(NUM_CLASSES)}


def dataset_meta(data_dir: str) -> dict:
    """Lightweight metadata for the generate script (loads or builds cache)."""
    data = load_base_dataset(data_dir, force=False)
    y = data.y.view(-1).long()
    deg = data.edge_index[0].bincount(minlength=int(data.num_nodes)).float()
    counts = _class_counts(y)
    n_pos = int((y > 0).sum().item())
    n = int(data.num_nodes)
    y_tr = y[data.train_mask]
    n_tr = int(y_tr.numel())
    n_pos_tr = int((y_tr > 0).sum().item())
    return {
        "registry_key": REGISTRY_KEY,
        "degree": DEGREE,
        "num_classes": NUM_CLASSES,
        "n_motifs_c3": M3,
        "n_motifs_c4": M4,
        "n_motifs_c5": M5,
        "class_counts": counts,
        "n_edges": motif_er.num_undirected_edges(data.edge_index),
        "mean_degree": float(deg.mean().item()),
        "mean_degree_pos": float(deg[y > 0].mean().item()) if bool((y > 0).any()) else float("nan"),
        "mean_degree_neg": float(deg[y == 0].mean().item()) if bool((y == 0).any()) else float("nan"),
        "cache": str(Path(data_dir) / REGISTRY_KEY / "processed" / "data.pt"),
        "n": float(n),
        "n_pos": float(n_pos),
        "n_neg": float(n - n_pos),
        "pos_rate": float(n_pos / n) if n else 0.0,
        "n_train": float(n_tr),
        "n_pos_train": float(n_pos_tr),
        "n_neg_train": float(n_tr - n_pos_tr),
        "pos_rate_train": float(n_pos_tr / n_tr) if n_tr else 0.0,
        "n_val": int(data.val_mask.sum().item()),
        "n_test": int(data.test_mask.sum().item()),
        "split": f"{N_TRAIN}/{N_VAL}/{N_TEST}",
        "has_node_features": HAS_NODE_FEATURES,
    }
