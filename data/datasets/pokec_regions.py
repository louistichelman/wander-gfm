"""Pokec Regions (GraphLand) dataset definition."""

from torch_geometric.data import Data

from ._graphland_loader import load_graphland_rl

NAME = "pokec-regions"
HAS_NODE_FEATURES = True
HAS_NODE_LABELS = True
HAS_PREDEFINED_NODE_SPLIT = True
HAS_PREDEFINED_EDGE_SPLIT = False


def load_base_dataset(
    data_dir: str,
    *,
    graphland_different_transform: bool = False,
    graphland_categorical_as_ordinals: bool = False,
) -> Data:
    """Load raw pokec-regions (GraphLand, RL split)."""
    return load_graphland_rl(
        data_dir,
        NAME,
        graphland_different_transform=graphland_different_transform,
        graphland_categorical_as_ordinals=graphland_categorical_as_ordinals,
    )
