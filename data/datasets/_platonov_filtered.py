"""Load Platonov et al. (ICLR 2023) filtered Wikipedia graphs.

Squirrel and Chameleon Geom-GCN graphs contain large groups of duplicate nodes
(same original traffic target and outgoing neighborhood, no incoming edges).
The filtered ``.npz`` files keep only the non-duplicate nodes and the 10
Geom-GCN splits with those nodes removed.

Source: https://github.com/yandex-research/heterophilous-graphs
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, download_url
from torch_geometric.utils import coalesce

from ..nc_splits import as_nodes_by_splits

_PLATONOV_NPZ_URL = (
    "https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/"
    "main/data/{npz_name}"
)


def load_platonov_filtered_wikipedia(
    data_dir: str,
    *,
    folder_name: str,
    npz_name: str,
) -> Data:
    """Load a filtered Squirrel/Chameleon graph as a PyG ``Data`` object.

    Official Geom-GCN masks stay 2-D ``[num_nodes, 10]``; ``DataSet`` selects
    ``nc_split_index``.

    Args:
        data_dir: Root raw-data directory (``SquirrelFiltered/`` etc. is created
            underneath).
        folder_name: Subdirectory under ``data_dir`` (matches dataset ``NAME``).
        npz_name: File name on GitHub, e.g. ``squirrel_filtered.npz``.
    """
    data_path = Path(data_dir) / folder_name
    data_path.mkdir(parents=True, exist_ok=True)
    npz_path = data_path / npz_name
    if not npz_path.is_file():
        download_url(_PLATONOV_NPZ_URL.format(npz_name=npz_name), str(data_path))

    raw = np.load(npz_path)
    x = torch.from_numpy(np.asarray(raw["node_features"])).float()
    y = torch.from_numpy(np.asarray(raw["node_labels"])).long()
    edges = torch.from_numpy(np.asarray(raw["edges"])).long()
    if edges.ndim != 2 or edges.size(1) != 2:
        raise ValueError(
            f"{npz_name}: expected edges of shape [E, 2], got {tuple(edges.shape)}"
        )
    edge_index = coalesce(edges.t().contiguous(), num_nodes=int(x.size(0)))

    n_nodes = int(x.size(0))
    train_mask = as_nodes_by_splits(
        torch.from_numpy(np.asarray(raw["train_masks"])).bool(), n_nodes
    )
    val_mask = as_nodes_by_splits(
        torch.from_numpy(np.asarray(raw["val_masks"])).bool(), n_nodes
    )
    test_mask = as_nodes_by_splits(
        torch.from_numpy(np.asarray(raw["test_masks"])).bool(), n_nodes
    )

    data = Data(
        x=x,
        y=y,
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=n_nodes,
    )
    return data
