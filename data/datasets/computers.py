"""Amazon Computers dataset definition."""

from pathlib import Path

from torch_geometric.data import Data
from torch_geometric.datasets import Amazon

# Dataset metadata
NAME = "Computers"
HAS_NODE_FEATURES = True
HAS_NODE_LABELS = True
HAS_PREDEFINED_NODE_SPLIT = False
HAS_PREDEFINED_EDGE_SPLIT = False


def load_base_dataset(data_dir: str) -> Data:
    """Load raw Amazon Computers co-purchase network.

    Args:
        data_dir: Directory to store/load the dataset.

    Returns:
        Raw PyG Data object.
    """
    data_path = Path(data_dir) / NAME
    dataset = Amazon(root=str(data_path), name=NAME)
    data = dataset[0]

    if not hasattr(data, "num_nodes") or data.num_nodes is None:
        data.num_nodes = (
            data.x.size(0)
            if hasattr(data, "x") and data.x is not None
            else data.edge_index.max().item() + 1
        )

    return data
