"""Shared loading for GraphLand benchmark datasets (PyG GraphLandDataset, RL split)."""

from pathlib import Path

from torch_geometric.data import Data

try:
    from torch_geometric.datasets import GraphLandDataset
except ImportError:  # PyPI builds may not include GraphLand yet
    from .graphland_dataset import GraphLandDataset

from ._graphland_graphpfn_raw import (
    build_graphland_transductive_data,
    load_graphland_raw_data,
)

ROOT_FOLDER = "GraphLand"
SPLIT = "RL"


def _ensure_graphland_raw_downloaded(root: Path, name: str) -> Path:
    raw_data_dir = root / name / "raw" / name
    if (raw_data_dir / "features.csv").exists():
        return raw_data_dir

    loader = GraphLandDataset.__new__(GraphLandDataset)
    loader.root = str(root)
    loader.name = name
    loader.download()
    if not (raw_data_dir / "features.csv").exists():
        raise FileNotFoundError(
            f"GraphLand raw files for {name!r} not found under {raw_data_dir}"
        )
    return raw_data_dir


def _load_graphland_rl_graphpfn_raw(data_dir: str, name: str) -> Data:
    """Load GraphPFN-aligned raw GraphLand features without processed-cache reuse."""
    root = Path(data_dir) / ROOT_FOLDER
    raw_data_dir = _ensure_graphland_raw_downloaded(root, name)
    raw_data = load_graphland_raw_data(raw_data_dir, SPLIT)
    return build_graphland_transductive_data(raw_data, to_undirected=True)


def load_graphland_rl(
    data_dir: str,
    name: str,
    *,
    graphland_different_transform: bool = False,
    graphland_categorical_as_ordinals: bool = False,
) -> Data:
    """Load a single-graph GraphLand classification dataset with the RL split."""
    if graphland_different_transform or graphland_categorical_as_ordinals:
        return _load_graphland_rl_graphpfn_raw(data_dir, name)

    root = Path(data_dir) / ROOT_FOLDER
    dataset = GraphLandDataset(
        root=str(root),
        name=name,
        split=SPLIT,
        numerical_features_transform="default",
        fraction_features_transform="default",
        categorical_features_transform="one_hot_encoding",
    )

    data = dataset[0]
    if not hasattr(data, "num_nodes") or data.num_nodes is None:
        data.num_nodes = int(data.x.size(0))
    return data
