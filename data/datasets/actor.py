"""Actor dataset definition."""

from pathlib import Path
from torch_geometric.datasets import Actor
from torch_geometric.data import Data

# Dataset metadata
NAME = "Actor"
HAS_NODE_FEATURES = True
HAS_NODE_LABELS = True
HAS_PREDEFINED_NODE_SPLIT = True
HAS_PREDEFINED_EDGE_SPLIT = False
NUM_PREDEFINED_NODE_SPLITS = 10


def load_base_dataset(data_dir: str) -> Data:
    """Load raw Actor dataset.
    
    Args:
        data_dir: Directory to store/load the dataset.
        
    Returns:
        Raw PyG Data object.
    """
    data_path = Path(data_dir) / NAME
    dataset = Actor(root=str(data_path))
    data = dataset[0]
    
    # Ensure num_nodes is set
    if not hasattr(data, 'num_nodes') or data.num_nodes is None:
        data.num_nodes = data.x.size(0) if hasattr(data, 'x') and data.x is not None else data.edge_index.max().item() + 1
    
    # Official masks are [num_nodes, 10]; DataSet selects nc_split_index.
    return data
