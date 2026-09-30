"""Pokec Regions 100k BFS subsample with only the 10 largest classes labeled.

Classes outside the top 10 are unlabeled and removed from train / val / test;
those nodes stay in the graph. The unlabeled 100k cache is an implementation
detail of ``pokec_regions_100k``, not a registered dataset.
"""

from __future__ import annotations

from torch_geometric.data import Data

from . import pokec_regions_100k as _base

NAME = "pokec-regions-100k-top10"
HAS_NODE_FEATURES = True
HAS_NODE_LABELS = True
HAS_PREDEFINED_NODE_SPLIT = True
HAS_PREDEFINED_EDGE_SPLIT = False

REGISTRY_KEY = "POKEC_REGIONS_100K_TOP10"
KEEP_TOP_CLASSES = 10


def load_base_dataset(
    data_dir: str,
    seed: int = _base.SEED,
    *,
    force: bool = False,
    target_nodes: int = _base.TARGET_NODES,
    class_rank: int = _base.CLASS_RANK,
    graphland_different_transform: bool = False,
    graphland_categorical_as_ordinals: bool = False,
) -> Data:
    """Load the 100k BFS graph with only the top-10 classes labeled."""
    return _base.load_base_dataset(
        data_dir,
        seed=seed,
        force=force,
        target_nodes=target_nodes,
        class_rank=class_rank,
        keep_top_classes=KEEP_TOP_CLASSES,
        graphland_different_transform=graphland_different_transform,
        graphland_categorical_as_ordinals=graphland_categorical_as_ordinals,
    )
