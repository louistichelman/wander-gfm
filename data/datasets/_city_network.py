"""Shared loader for the City-Networks node-classification datasets.

Three large real-world road networks (Paris, Shanghai, LA) introduced in
`"Towards Quantifying Long-Range Interactions in Graph Machine Learning: a
Large Graph Dataset and a Measurement" <https://arxiv.org/abs/2503.09008>`_.
Nodes are road junctions, edges are undirected road segments, and the label is
each node's eccentricity score (approximated from its 16-hop neighborhood)
bucketed into 10 quantiles -- a long-range transductive classification task.
Each city ships a single official train/val/test split (1-D masks), so no
split-protocol wiring beyond split 0 is needed.

Requires torch_geometric>=2.7.0 (``CityNetwork`` was added in that release).
Source: https://github.com/LeonResearch/City-Networks
"""

from __future__ import annotations

from pathlib import Path

from torch_geometric.data import Data
from torch_geometric.datasets import CityNetwork

FOLDER_NAME = "CityNetwork"


def load_city_network(data_dir: str, *, name: str) -> Data:
    """Load one City-Networks city as a PyG ``Data`` object.

    Args:
        data_dir: Root raw-data directory. A shared ``CityNetwork/`` folder is
            created underneath (the dataset class itself nests further by
            city name, e.g. ``CityNetwork/paris/raw``).
        name: City variant -- ``"paris"``, ``"shanghai"``, or ``"la"``.
    """
    data_path = Path(data_dir) / FOLDER_NAME
    dataset = CityNetwork(root=str(data_path), name=name)
    data = dataset[0]

    if not hasattr(data, "num_nodes") or data.num_nodes is None:
        data.num_nodes = data.x.size(0)

    return data
