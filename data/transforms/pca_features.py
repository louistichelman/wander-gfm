"""Transform to reduce feature dimensionality using PCA."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data


def _features_matrix_for_graphpfn_pca(data: Data) -> np.ndarray:
    """Build the feature matrix GraphPFN concatenates before ``apply_d_reduction``.

    For homogeneous graphs this is ``data.x``. For GraphLand graphs with column-type
    masks, categorical columns are one-hot encoded before concatenation (num, frac,
    cat) — matching ``lib.graph.data.apply_d_reduction``.
    """
    x = data.x.detach().cpu().numpy().astype(np.float32, copy=False)
    num_mask = getattr(data, "x_numerical_mask", None)
    if num_mask is None:
        return x

    num_mask = num_mask.bool().cpu().numpy()
    frac_mask = getattr(data, "x_fraction_mask", None)
    frac_mask = (
        frac_mask.bool().cpu().numpy()
        if frac_mask is not None
        else np.zeros(x.shape[1], dtype=bool)
    )
    cat_mask = getattr(data, "x_categorical_mask", None)
    cat_mask = (
        cat_mask.bool().cpu().numpy()
        if cat_mask is not None
        else np.zeros(x.shape[1], dtype=bool)
    )

    parts: list[np.ndarray] = []
    if num_mask.any():
        parts.append(x[:, num_mask])
    if frac_mask.any():
        parts.append(x[:, frac_mask])
    if cat_mask.any():
        cat_block = x[:, cat_mask]
        enc = OneHotEncoder(
            drop="if_binary",
            sparse_output=False,
            handle_unknown="ignore",
            dtype=np.float32,
        )
        enc.fit(cat_block)
        parts.append(enc.transform(cat_block))
    return np.concatenate(parts, axis=1) if parts else x


class PCAFeatures:
    """Reduce node feature dimensionality using PCA (GraphPFN-compatible).

    Uses ``sklearn.decomposition.PCA`` like GraphPFN ``apply_d_reduction``:
    ``PCA(n_components=target_dim).fit_transform(features)`` on all nodes.

    If the input has ``target_dim`` or fewer columns after optional GraphLand
    one-hot expansion, features are left unchanged.
    """

    def __init__(self, target_dim: int = 512):
        self.target_dim = target_dim

    def __call__(self, data: Data) -> Data:
        if not hasattr(data, "x") or data.x is None:
            return data

        # GraphPFN counts ordinal categorical columns toward ``d``; one-hot expansion
        # happens only when reduction is actually applied.
        if data.x.size(1) <= self.target_dim:
            return data

        features = _features_matrix_for_graphpfn_pca(data)
        if features.shape[1] <= self.target_dim:
            return data

        if not np.isfinite(features).all():
            return data

        reduced = PCA(n_components=self.target_dim).fit_transform(features)
        data.x = torch.from_numpy(reduced).to(data.x.device, dtype=torch.float32)

        # Merged PCA output is a single numerical block (GraphPFN clears typed keys).
        for attr in ("x_numerical_mask", "x_fraction_mask", "x_categorical_mask"):
            if hasattr(data, attr):
                setattr(data, attr, None)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(target_dim={self.target_dim})"
