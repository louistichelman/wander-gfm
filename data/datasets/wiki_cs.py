"""WikiCS dataset definition."""

from pathlib import Path

from torch_geometric.data import Data
from torch_geometric.datasets import WikiCS

# Dataset metadata
NAME = "Wiki_CS"
HAS_NODE_FEATURES = True
HAS_NODE_LABELS = True
HAS_PREDEFINED_NODE_SPLIT = True
HAS_PREDEFINED_EDGE_SPLIT = False
NUM_PREDEFINED_NODE_SPLITS = 20


def load_base_dataset(data_dir: str) -> Data:
    """Load raw WikiCS dataset (20 train/val splits, shared test mask).

    Args:
        data_dir: Directory to store/load the dataset.

    Returns:
        Raw PyG Data object with 2-D train/val masks; DataSet selects nc_split_index.
    """
    data_path = Path(data_dir) / NAME
    dataset = WikiCS(root=str(data_path))
    data = dataset[0]

    if not hasattr(data, "num_nodes") or data.num_nodes is None:
        data.num_nodes = (
            data.x.size(0)
            if hasattr(data, "x") and data.x is not None
            else data.edge_index.max().item() + 1
        )

    # train/val are [num_nodes, 20]; test_mask is 1-D (shared). DataSet selects the split.
    return data
