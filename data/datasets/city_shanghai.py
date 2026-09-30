"""Shanghai road network (City-Networks) dataset definition.

183,917 nodes / 262,092 edges / 37 features / 10 classes.
"""

from torch_geometric.data import Data

from ._city_network import load_city_network

NAME = "City-Shanghai"
HAS_NODE_FEATURES = True
HAS_NODE_LABELS = True
HAS_PREDEFINED_NODE_SPLIT = True
HAS_PREDEFINED_EDGE_SPLIT = False


def load_base_dataset(data_dir: str) -> Data:
    """Load raw Shanghai City-Networks dataset (single official split)."""
    return load_city_network(data_dir, name="shanghai")
