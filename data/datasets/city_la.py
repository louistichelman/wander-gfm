"""LA road network (City-Networks) dataset definition.

240,587 nodes / 341,523 edges / 37 features / 10 classes.
"""

from torch_geometric.data import Data

from ._city_network import load_city_network

NAME = "City-LA"
HAS_NODE_FEATURES = True
HAS_NODE_LABELS = True
HAS_PREDEFINED_NODE_SPLIT = True
HAS_PREDEFINED_EDGE_SPLIT = False


def load_base_dataset(data_dir: str) -> Data:
    """Load raw LA City-Networks dataset (single official split)."""
    return load_city_network(data_dir, name="la")
