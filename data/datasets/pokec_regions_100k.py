"""Pokec Regions 100k BFS subsample helper (used by ``POKEC_REGIONS_100K_TOP10``).

Starts at a random train node of the third-largest region class, then keeps
the first 100k nodes in BFS order and the induced subgraph. Original RL masks
are restricted to the kept nodes. Appearing class ids are remapped to ``0..K-1``.

Not a registered dataset. Cached under ``raw_data/POKEC_REGIONS_100K/processed/``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Data

from . import pokec_regions
from ..bfs_subsample import restrict_to_top_classes, subsample_bfs

NAME = "pokec-regions-100k"
HAS_NODE_FEATURES = True
HAS_NODE_LABELS = True
HAS_PREDEFINED_NODE_SPLIT = True
HAS_PREDEFINED_EDGE_SPLIT = False

REGISTRY_KEY = "POKEC_REGIONS_100K"
TARGET_NODES = 100_000
CLASS_RANK = 2
SEED = 0


def cache_path(
    data_dir: str,
    *,
    seed: int = SEED,
    target_nodes: int = TARGET_NODES,
    class_rank: int = CLASS_RANK,
    keep_top_classes: int | None = None,
    graphland_categorical_as_ordinals: bool = False,
) -> Path:
    name = f"data_seed{int(seed)}_n{int(target_nodes)}_rank{int(class_rank)}"
    if keep_top_classes is not None:
        name += f"_top{int(keep_top_classes)}"
    if graphland_categorical_as_ordinals:
        name += "_ord"
    return Path(data_dir) / REGISTRY_KEY / "processed" / f"{name}.pt"


def load_base_dataset(
    data_dir: str,
    seed: int = SEED,
    *,
    force: bool = False,
    target_nodes: int = TARGET_NODES,
    class_rank: int = CLASS_RANK,
    keep_top_classes: int | None = None,
    graphland_different_transform: bool = False,
    graphland_categorical_as_ordinals: bool = False,
) -> Data:
    """Load the cached 100k BFS subsample, building it from full Pokec if needed.

    ``keep_top_classes`` keeps labels/splits only for the largest K classes
    in the subsample; other nodes stay in the graph but unlabeled.
    """
    path = cache_path(
        data_dir,
        seed=seed,
        target_nodes=target_nodes,
        class_rank=class_rank,
        keep_top_classes=keep_top_classes,
        graphland_categorical_as_ordinals=graphland_categorical_as_ordinals,
    )
    if path.is_file() and not force:
        return torch.load(path, map_location="cpu", weights_only=False)

    if keep_top_classes is not None:
        base = load_base_dataset(
            data_dir,
            seed=seed,
            force=force,
            target_nodes=target_nodes,
            class_rank=class_rank,
            keep_top_classes=None,
            graphland_different_transform=graphland_different_transform,
            graphland_categorical_as_ordinals=graphland_categorical_as_ordinals,
        )
        data = restrict_to_top_classes(base, int(keep_top_classes))
    else:
        full = pokec_regions.load_base_dataset(
            data_dir,
            graphland_different_transform=graphland_different_transform,
            graphland_categorical_as_ordinals=graphland_categorical_as_ordinals,
        )
        data = subsample_bfs(
            full,
            target_nodes=target_nodes,
            class_rank=class_rank,
            seed=seed,
        ).data

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    if not path.is_file():
        raise RuntimeError(f"Failed to write Pokec 100k cache: {path}")
    return data
