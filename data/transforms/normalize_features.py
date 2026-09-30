"""Transform to normalize node features."""

from typing import Optional

import torch
from torch_geometric.data import Data


class NormalizeFeatures:
    """Normalize node features.

    Two modes are supported:

    - ``"col_zscore"`` (default): column-wise z-score using statistics computed on
      all nodes when ``train_mask`` is ``None`` (transductive), or on training nodes
      only when a valid ``train_mask`` is provided. The scaling is applied to all
      nodes so downstream models see consistent units.
    - ``"row_l2"``: per-row unit-L2 normalization. Each node's feature
      vector is rescaled to unit L2 norm; ``train_mask`` is ignored.
    - ``"row_l1"``: per-row L1 normalization:
      ``x / (sum(x) + 0.01)``; ``train_mask`` is ignored.

    Args:
        mode: One of ``"col_zscore"``, ``"row_l2"``, or ``"row_l1"``.
        train_mask: Optional 1D boolean tensor of length ``num_nodes`` selecting
            the rows used to compute column statistics. Only used when
            ``mode="col_zscore"``.
        clip_value: Optional symmetric clip applied after normalization
            (GraphPFN/LimiX clip normalized features to ``+-100``).
    """

    def __init__(
        self,
        mode: str = "col_zscore",
        train_mask: Optional[torch.Tensor] = None,
        clip_value: Optional[float] = None,
    ) -> None:
        if mode not in ("col_zscore", "row_l2", "row_l1"):
            raise ValueError(
                f"NormalizeFeatures: unknown mode {mode!r}; "
                "expected 'col_zscore', 'row_l2', or 'row_l1'."
            )
        self.mode = mode
        self.train_mask = train_mask
        self.clip_value = clip_value

    def __call__(self, data: Data) -> Data:
        """Apply the transform to normalize node features.

        Args:
            data: PyG Data object with node features ``x``.

        Returns:
            Data object with normalized features.
        """
        if not hasattr(data, "x") or data.x is None:
            return data

        x = data.x.float()

        if self.mode == "row_l2":
            norm = x.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
            data.x = x / norm
            return data

        if self.mode == "row_l1":
            rowsum = x.sum(dim=1, keepdim=True) + 0.01
            data.x = x / rowsum
            return data

        # col_zscore
        train_mask = self.train_mask
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

        mean = rows.mean(dim=0, keepdim=True)
        std = rows.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-12)
        x = (x - mean) / std
        if self.clip_value is not None:
            x = x.clamp(min=-self.clip_value, max=self.clip_value)
        data.x = x
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mode={self.mode!r})"
