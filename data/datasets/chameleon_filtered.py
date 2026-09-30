"""Chameleon filtered (Platonov et al., ICLR 2023) dataset definition."""

from torch_geometric.data import Data

from ._platonov_filtered import load_platonov_filtered_wikipedia

NAME = "ChameleonFiltered"
HAS_NODE_FEATURES = True
HAS_NODE_LABELS = True
HAS_PREDEFINED_NODE_SPLIT = True
HAS_PREDEFINED_EDGE_SPLIT = False
NUM_PREDEFINED_NODE_SPLITS = 10


def load_base_dataset(data_dir: str) -> Data:
    """Load filtered Chameleon (duplicate nodes removed, 10 Geom-GCN splits)."""
    return load_platonov_filtered_wikipedia(
        data_dir,
        folder_name=NAME,
        npz_name="chameleon_filtered.npz",
    )
