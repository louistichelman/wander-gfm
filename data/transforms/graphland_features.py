"""Per-type GraphLand feature transforms (GraphPFN-style)."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from torch_geometric.data import Data

# Matches GraphPFN ``transform_num_features`` quantile settings (seed=0 in ICL eval).
_GRAPHPFN_QUANTILE_SUBSAMPLE = 1_000_000_000
_GRAPHPFN_QUANTILE_JITTER_STD = 1e-5


def _fit_rows(x: torch.Tensor, train_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if (
        train_mask is not None
        and train_mask.dim() == 1
        and train_mask.dtype == torch.bool
        and train_mask.numel() == x.size(0)
        and train_mask.any()
    ):
        return x[train_mask].detach().cpu()
    return x.detach().cpu()


def _graphpfn_n_quantiles(n_rows: int) -> int:
    return max(min(n_rows // 30, 1000), 10)


def _impute_nans_block(
    block: torch.Tensor,
    *,
    strategy: str = "most_frequent",
) -> torch.Tensor:
    """Impute NaNs using GraphPFN ``impute_nans`` (transductive: fit on all nodes)."""
    arr = block.detach().cpu().numpy()
    if not np.isnan(arr).any():
        return block

    imputer = SimpleImputer(strategy=strategy)
    imputer.fit(arr)
    out = imputer.transform(arr)
    return torch.from_numpy(out).to(block.device, dtype=block.dtype)


def _quantile_transform_block(
    block: torch.Tensor,
    output_distribution: Literal["normal", "uniform"],
    *,
    seed: int = 0,
) -> torch.Tensor:
    """Apply GraphPFN-style quantile transform (transductive: fit on all nodes)."""
    arr = block.detach().cpu().numpy()
    n_seen = arr.shape[0]
    fit_arr = arr + np.random.RandomState(seed).normal(
        0.0, _GRAPHPFN_QUANTILE_JITTER_STD, arr.shape
    ).astype(arr.dtype, copy=False)

    qt = QuantileTransformer(
        n_quantiles=_graphpfn_n_quantiles(n_seen),
        output_distribution=output_distribution,
        subsample=_GRAPHPFN_QUANTILE_SUBSAMPLE,
        random_state=seed,
        copy=False,
    )
    qt.fit(fit_arr)
    out = qt.transform(arr)
    return _impute_nans_block(torch.from_numpy(out).to(block.device, dtype=block.dtype))


def apply_graphland_per_type_transforms(
    data: Data,
    *,
    train_mask: Optional[torch.Tensor] = None,
    categorical_as_ordinals: bool = True,
    seed: int = 0,
) -> Data:
    """Apply quantile transforms to numerical/fraction columns on GraphLand data."""
    if not hasattr(data, "x") or data.x is None:
        return data

    num_mask = getattr(data, "x_numerical_mask", None)
    frac_mask = getattr(data, "x_fraction_mask", None)
    cat_mask = getattr(data, "x_categorical_mask", None)
    if num_mask is None or frac_mask is None or cat_mask is None:
        return data

    num_cols = num_mask.bool().nonzero(as_tuple=True)[0]
    frac_cols = frac_mask.bool().nonzero(as_tuple=True)[0]
    cat_cols = cat_mask.bool().nonzero(as_tuple=True)[0]

    parts: list[torch.Tensor] = []
    num_flags: list[torch.Tensor] = []
    frac_flags: list[torch.Tensor] = []
    cat_flags: list[torch.Tensor] = []

    def _append(block: torch.Tensor, kind: str) -> None:
        w = block.size(1)
        parts.append(block)
        num_flags.append(torch.zeros(w, dtype=torch.bool))
        frac_flags.append(torch.zeros(w, dtype=torch.bool))
        cat_flags.append(torch.zeros(w, dtype=torch.bool))
        if kind == "num":
            num_flags[-1][:] = True
        elif kind == "frac":
            frac_flags[-1][:] = True
        else:
            cat_flags[-1][:] = True

    if num_cols.numel() > 0:
        block = _quantile_transform_block(
            data.x[:, num_cols].float(),
            "normal",
            seed=seed,
        )
        _append(block, "num")

    if frac_cols.numel() > 0:
        block = _quantile_transform_block(
            data.x[:, frac_cols].float(),
            "uniform",
            seed=seed,
        )
        _append(block, "frac")

    if cat_cols.numel() > 0:
        block = data.x[:, cat_cols].float()
        if categorical_as_ordinals:
            _append(block, "cat")
        else:
            enc = OneHotEncoder(
                drop="if_binary",
                sparse_output=False,
                handle_unknown="ignore",
            )
            enc.fit(_fit_rows(block, train_mask).numpy())
            oh = torch.from_numpy(enc.transform(block.cpu().numpy())).to(
                block.device, dtype=block.dtype
            )
            _append(oh, "cat")

    if not parts:
        return data

    data.x = torch.cat(parts, dim=1)
    data.x_numerical_mask = torch.cat(num_flags)
    data.x_fraction_mask = torch.cat(frac_flags)
    data.x_categorical_mask = torch.cat(cat_flags)
    return data
