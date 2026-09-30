"""Drop columns that are constant on training nodes."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch_geometric.data import Data


def drop_constant_features(
    x: torch.Tensor,
    train_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(x[:, keep_mask], keep_mask)`` dropping train-constant columns."""
    if x.ndim != 2:
        raise ValueError(f"Expected 2D feature tensor, got shape {tuple(x.shape)}")

    if x.size(1) == 0:
        return x, torch.zeros(0, dtype=torch.bool, device=x.device)

    if (
        train_mask is not None
        and train_mask.dim() == 1
        and train_mask.dtype == torch.bool
        and train_mask.numel() == x.size(0)
        and train_mask.any()
    ):
        rows = x[train_mask]
    else:
        rows = x

    keep_mask = torch.tensor(
        [rows[:, i].unique().numel() > 1 for i in range(x.size(1))],
        dtype=torch.bool,
        device=x.device,
    )
    if not keep_mask.any():
        keep_mask[0] = True
    return x[:, keep_mask], keep_mask


class DropConstantTrainFeatures:
    """Drop feature columns that are constant on training nodes."""

    def __init__(self, train_mask: Optional[torch.Tensor] = None) -> None:
        self.train_mask = train_mask

    def __call__(self, data: Data) -> Data:
        if not hasattr(data, "x") or data.x is None or data.x.size(1) == 0:
            return data
        x, keep_mask = drop_constant_features(data.x, self.train_mask)
        data.x = x
        data.feature_keep_mask = keep_mask
        for attr in ("x_numerical_mask", "x_fraction_mask", "x_categorical_mask"):
            if hasattr(data, attr) and getattr(data, attr) is not None:
                setattr(data, attr, getattr(data, attr)[keep_mask.cpu()])
        return data
