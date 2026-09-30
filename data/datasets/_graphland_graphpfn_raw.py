"""GraphPFN-aligned raw GraphLand feature loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import ToUndirected


def drop_constant_feature_columns(features_df) -> tuple[Any, list[str]]:
    """Drop columns with a single unique value (GraphPFN ``_load_graphland_data``)."""
    remaining = features_df.loc[:, features_df.apply(lambda s: s.nunique()) != 1]
    return remaining, list(remaining.columns)


def split_graphland_feature_blocks(
    features_df,
    info: dict,
    columns_remained: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split cleaned features into numerical, fraction, and categorical blocks."""
    frac_names = [
        name for name in info["fraction_features_names"] if name in columns_remained
    ]
    frac_features = (
        features_df.loc[:, frac_names].to_numpy(dtype=np.float32)
        if frac_names
        else np.empty((len(features_df), 0), dtype=np.float32)
    )

    cat_names = [
        name for name in info["categorical_features_names"] if name in columns_remained
    ]
    cat_features = (
        features_df.loc[:, cat_names].to_numpy(dtype=np.int32)
        if cat_names
        else np.empty((len(features_df), 0), dtype=np.int32)
    )

    num_names = [
        name
        for name in info["numerical_features_names"]
        if name not in frac_names and name in columns_remained
    ]
    num_features = (
        features_df.loc[:, num_names].to_numpy(dtype=np.float32)
        if num_names
        else np.empty((len(features_df), 0), dtype=np.float32)
    )

    if frac_features.size > 0:
        assert not np.isnan(frac_features).any()

    return num_features, frac_features, cat_features


def apply_graphpfn_graphland_feature_cleanup(
    num_features: np.ndarray,
    frac_features: np.ndarray,
    cat_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror GraphPFN ``load_data`` post-processing for GraphLand features."""
    if cat_features.size > 0:
        keep = np.array([len(np.unique(col)) > 1 for col in cat_features.T], dtype=bool)
        assert keep.any(), "GraphLand categorical features cannot all be constant."
        cat_features = cat_features[:, keep]

    if frac_features.size > 0:
        bin_mask = np.array(
            [np.all((col == 0.0) | (col == 1.0)) for col in frac_features.T],
            dtype=bool,
        )
        if bin_mask.any():
            bin_block = frac_features[:, bin_mask].astype(np.float32)
            cat_features = (
                np.concatenate([cat_features, bin_block], axis=1)
                if cat_features.size > 0
                else bin_block
            )
            frac_features = (
                frac_features[:, :0]
                if bin_mask.all()
                else frac_features[:, ~bin_mask]
            )

    return num_features, frac_features, cat_features


def load_graphland_raw_data(raw_data_dir: str | Path, split: str) -> dict[str, Any]:
    """Load GraphLand CSV/YAML files and apply GraphPFN raw feature cleanup."""
    import pandas as pd

    from .graphland_dataset import _load_yaml

    raw_data_dir = Path(raw_data_dir)
    info = _load_yaml(str(raw_data_dir / "info.yaml"))

    features_df = pd.read_csv(
        raw_data_dir / "features.csv",
        index_col=0,
    ).astype(np.float32)
    features_df, columns_remained = drop_constant_feature_columns(features_df)
    num_features, frac_features, cat_features = split_graphland_feature_blocks(
        features_df, info, columns_remained
    )
    num_features, frac_features, cat_features = apply_graphpfn_graphland_feature_cleanup(
        num_features, frac_features, cat_features
    )

    targets_df = pd.read_csv(raw_data_dir / "targets.csv", index_col=0)
    targets = targets_df[info["target_name"]].values.astype(np.float32)

    masks_df = pd.read_csv(
        raw_data_dir / f"split_masks_{split[:2]}.csv",
        index_col=0,
    )
    masks = {k: np.array(v, dtype=bool) for k, v in masks_df.to_dict("list").items()}

    edges_df = pd.read_csv(raw_data_dir / "edgelist.csv")
    edges = edges_df.values

    return {
        "info": info,
        "num_features": num_features,
        "cat_features": cat_features,
        "frac_features": frac_features,
        "targets": targets,
        "masks": masks,
        "edges": edges,
    }


def build_graphland_transductive_data(
    raw_data: dict[str, Any],
    *,
    to_undirected: bool = True,
) -> Data:
    """Build a transductive GraphLand ``Data`` object without sklearn transforms."""
    num_features = raw_data["num_features"]
    frac_features = raw_data["frac_features"]
    cat_features = raw_data["cat_features"]

    features = np.concatenate(
        [num_features, frac_features, cat_features],
        axis=1,
    )
    features = torch.from_numpy(features).float()

    num_mask = torch.zeros(features.shape[1], dtype=torch.bool)
    num_mask[: num_features.shape[1]] = True

    frac_mask = torch.zeros(features.shape[1], dtype=torch.bool)
    if cat_features.shape[1] > 0:
        frac_mask[num_features.shape[1] : -cat_features.shape[1]] = True
    else:
        frac_mask[num_features.shape[1] :] = True

    cat_mask = torch.zeros(features.shape[1], dtype=torch.bool)
    if cat_features.shape[1] > 0:
        cat_mask[-cat_features.shape[1] :] = True

    targets = raw_data["targets"]
    labeled_mask = ~np.isnan(targets)
    targets = torch.from_numpy(targets).float()

    train_mask = torch.from_numpy(raw_data["masks"]["train"] & labeled_mask).bool()
    val_mask = torch.from_numpy(raw_data["masks"]["val"] & labeled_mask).bool()
    test_mask = torch.from_numpy(raw_data["masks"]["test"] & labeled_mask).bool()

    edge_index = torch.from_numpy(raw_data["edges"].T).long()

    data = Data(
        edge_index=edge_index,
        x=features,
        y=targets,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        x_numerical_mask=num_mask,
        x_fraction_mask=frac_mask,
        x_categorical_mask=cat_mask,
    )
    if to_undirected:
        data = ToUndirected()(data)
    if not hasattr(data, "num_nodes") or data.num_nodes is None:
        data.num_nodes = int(data.x.size(0))
    return data
