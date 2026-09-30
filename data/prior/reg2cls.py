"""Turn SCM regression outputs into classification features and labels.

Applies the categorical plan from ``SCMPrior`` (binning, balancing, optional
one-hot mixing of ``y`` into ``X``).
"""

from __future__ import annotations

import random
import warnings

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


def torch_nanstd(input, dim=None, keepdim=False, ddof=0, *, dtype=None) -> Tensor:
    """Calculates the standard deviation of a tensor, ignoring NaNs, using NumPy internally.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    dim : int or tuple[int], optional
        The dimension or dimensions to reduce. Defaults to None (reduce all dimensions).

    keepdim : bool, optional
        Whether the output tensor has `dim` retained or not. Defaults to False.

    ddof : int, optional
        Delta Degrees of Freedom.

    dtype : torch.dtype, optional
        The desired data type of returned tensor. Defaults to None.

    Returns
    -------
    Tensor
        The standard deviation.
    """
    device = input.device
    np_input = input.cpu().numpy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        std = np.nanstd(np_input, axis=dim, dtype=dtype, keepdims=keepdim, ddof=ddof)

    return torch.from_numpy(std).to(dtype=torch.float, device=device)


def standard_scaling(input: Tensor, clip_value: float = 100) -> Tensor:
    """Standardizes features by removing the mean and scaling to unit variance.

    NaNs are ignored in mean/std calculation.

    Parameters
    ----------
    input : Tensor
        Input tensor of shape (T, H), where T is sequence length, H is features.

    clip_value : float, optional, default=100
        The value to clip the standardized input to, preventing extreme outliers.

    Returns
    -------
    Tensor
        The standardized input, clipped between -clip_value and clip_value.
    """
    mean = torch.nanmean(input, dim=0)
    std = torch_nanstd(input, dim=0, ddof=1 if input.shape[0] > 1 else 0).clip(min=1e-6)
    scaled_input = (input - mean) / std

    return torch.clip(scaled_input, min=-clip_value, max=clip_value)


def outlier_removing(input: Tensor, threshold: float = 4.0) -> Tensor:
    """Clamps outliers in the input tensor based on a specified number of standard deviations (threshold).

    Parameters
    ----------
    input : Tensor
        Input tensor of shape (T, H).

    threshold : float, optional, default=4.0
        Number of standard deviations to use as the cutoff.

    Returns
    -------
    Tensor
        The tensor with outliers clamped.
    """
    # First stage: Identify outliers using initial statistics
    mean = torch.nanmean(input, dim=0)
    std = torch_nanstd(input, dim=0, ddof=1 if input.shape[0] > 1 else 0).clip(min=1e-6)
    cut_off = std * threshold
    lower, upper = mean - cut_off, mean + cut_off

    # Create mask for non-outlier, non-NaN values
    mask = (lower <= input) & (input <= upper) & ~torch.isnan(input)

    # Second pass using only non-outlier values for mean/std
    masked_input = torch.where(mask, input, torch.nan)
    masked_mean = torch.nanmean(masked_input, dim=0)
    masked_std = torch_nanstd(
        masked_input, dim=0, ddof=1 if input.shape[0] > 1 else 0
    ).clip(min=1e-6)

    # Handle cases where a column had <= 1 valid value after masking -> std is NaN or 0
    masked_mean = torch.where(torch.isnan(masked_mean), mean, masked_mean)
    masked_std = torch.where(torch.isnan(masked_std), torch.zeros_like(std), masked_std)

    # Recalculate cutoff with robust estimates
    cut_off = masked_std * threshold
    lower, upper = masked_mean - cut_off, masked_mean + cut_off

    # Replace NaN bounds with +/- inf
    lower = torch.nan_to_num(lower, nan=-torch.inf)
    upper = torch.nan_to_num(upper, nan=torch.inf)

    return input.clamp(min=lower, max=upper)


def permute_classes(input: Tensor) -> Tensor:
    """Label encoding and permute classes.

    Parameters
    ----------
    input : Tensor
        Target of shape (T,) containing class labels.

    Returns
    -------
    Tensor
        Target with potentially permuted labels (T,).
    """
    unique_vals, _ = torch.unique(input, return_inverse=True)
    num_classes = len(unique_vals)

    if num_classes <= 1:  # No permutation needed for single class
        return input

    # Ensure labels are encoded from 0 to num_classes-1
    indices = unique_vals.argsort()
    mapped = indices[torch.searchsorted(unique_vals, input)]

    # Randomly permute classes
    perm = torch.randperm(num_classes, device=input.device)
    permuted = perm[mapped]

    return permuted


class BalancedBinarize:
    """Binarizes the input based on its median value."""

    def __call__(self, input: Tensor) -> Tensor:
        """
        Parameters
        ----------
        input : Tensor
            Input of shape (T,).

        Returns
        -------
        Tensor
            Binarized output (0 or 1) of shape (T,).
        """
        return (input > torch.median(input)).float()


class MulticlassAssigner:
    """Transforms the input into discrete classes using rank-based or value-based thresholding.

    Input shape: (T,) -> Output shape: (T,)
    """

    def __init__(self, num_classes: int, mode: str = "rank", ordered_prob: float = 0.2):
        """
        Parameters
        ----------
        num_classes : int
            The target number of discrete classes to output.

        mode : str, default="rank"
            The method used to determine class boundaries:
            - "rank": Boundaries are randomly sampled from the input.
            - "value": Boundaries are randomly sampled from a normal distribution.

        ordered_prob : float, default=0.2
            Probability of keeping the natural class order.
        """
        if num_classes < 2:
            raise ValueError(
                "The number of classes must be at least 2 for MulticlassAssigner."
            )

        self.num_classes = num_classes
        self.ordered_prob = ordered_prob
        self.mode = mode

    def __call__(self, input: Tensor) -> Tensor:
        """
        Parameters
        ----------
        input : Tensor
            Input of shape (T,).

        Returns
        -------
        Tensor
            Class labels of shape (T,) with integer values [0, num_classes-1].
        """

        T = input.shape[0]
        device = input.device

        if self.mode == "rank":
            boundary_indices = torch.randint(
                0, T, (self.num_classes - 1,), device=device
            )
            boundaries = input[boundary_indices]
        elif self.mode == "value":
            boundaries = torch.randn(self.num_classes - 1, device=device)

        # Compare input tensor with boundaries and sum across the boundary dimension to get classes
        classes = (input.unsqueeze(-1) > boundaries.unsqueeze(0)).sum(dim=1)

        # Permute classes
        if random.random() > self.ordered_prob:
            classes = permute_classes(classes)

        # Reverse classes
        if random.random() > 0.5:
            classes = self.num_classes - 1 - classes

        return classes


class Reg2Cls:
    """Transforms a single regression dataset (features X, targets y) into a
    classification format through feature processing (categorical conversion,
    one-hot encoding, normalization) and target transformation.

    The categorical conversion plan is pre-computed in ``SCMPrior.get_batch()``
    and executed deterministically here.

    Parameters
    ----------
    hp : dict
        Configuration dictionary. Expected keys include:

        - ``num_classes`` (int): Number of target classes (0 for regression).
        - ``multiclass_type`` (str): ``'rank'`` or ``'value'``.
        - ``balanced`` (bool): Whether to enforce balanced binary classes.
        - ``multiclass_ordered_prob`` (float): Prob. of keeping natural class order for y.
        - ``cat_plan`` (list[dict]): Pre-computed categorical conversion plan.
          Each entry has ``num_cats`` (int) and ``is_onehot`` (bool).
        - ``x_is_onehot_y`` (bool): If True, set X to a random affine map of
          one_hot(permuted y) into ``x_onehot_output_dim`` features (still easy, not
          identical to y).
        - ``x_onehot_output_dim`` (int): Output width for the one-hot path; required
          when ``x_is_onehot_y`` is True (set by ``SCMPrior``).
        - ``x_onehot_noise_frac`` (float): After the affine map, fraction of feature
          dimensions per row (independently) that receive additive Gaussian noise
          (default ``0.2``). Set to ``0`` to disable.
        - ``permute_features`` (bool): Whether to shuffle column order.
        - ``permute_labels`` (bool): Whether to permute final class labels.
    """

    def __init__(self, hp: dict):
        self.hp = hp

        num_classes = self.hp["num_classes"]
        if num_classes == 0:
            self.class_assigner = None
        elif num_classes == 2 and self.hp.get("balanced", False):
            self.class_assigner = BalancedBinarize()
        elif num_classes >= 2:
            self.class_assigner = MulticlassAssigner(
                num_classes,
                mode=self.hp["multiclass_type"],
                ordered_prob=self.hp["multiclass_ordered_prob"],
            )
        else:
            raise ValueError(f"Invalid number of classes: {num_classes}")

    def __call__(self, X: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """Process a single dataset (X, y) according to the hyperparameters.

        Parameters
        ----------
        X : Tensor
            Features of shape ``(T, H)``.
        y : Tensor
            Continuous targets of shape ``(T,)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(X_processed, y_processed)`` where X has variable width and
            y contains integer class labels (or scaled regression targets).
        """
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Input shapes mismatch or incorrect dims. X: {X.shape}, y: {y.shape}"
            )

        # 1. Classify y first (needed before the x_is_onehot_y branch)
        y = standard_scaling(y.unsqueeze(-1)).squeeze(-1)
        if self.class_assigner is not None:
            y = self.class_assigner(y)
            if self.hp.get("permute_labels", True):
                y = permute_classes(y)

        # 2. Build X
        if self.hp.get("x_is_onehot_y", False):
            X = self._make_onehot_y(y)
        else:
            X, continuous_mask = self._num2cat(X)
            X = self._process_features(X, continuous_mask)

        return X.float(), y.float()

    def _make_onehot_y(self, y: Tensor) -> Tensor:
        """Easy features: one_hot(permuted y) then a random linear layer.

        The inner permutation decorrelates displayed labels from one-hot axes;
        ``F.linear`` maps into ``x_onehot_output_dim`` so X matches the sampled
        feature width used for non–one-hot datasets. Optionally adds independent
        Gaussian noise on a random fraction of features per row so same-class
        points are similar but not identical.
        """
        y_long = y.long()
        num_classes = int(y_long.max().item()) + 1
        permuted = permute_classes(y_long)
        one_hot = F.one_hot(permuted, num_classes).float()
        H = int(self.hp.get("x_onehot_output_dim", num_classes))
        dev = one_hot.device
        dtype = one_hot.dtype
        scale = float(self.hp.get("init_std", 1.0))
        weight = torch.randn(H, num_classes, device=dev, dtype=dtype) * scale
        bias = torch.randn(H, device=dev, dtype=dtype) * scale
        X = F.linear(one_hot, weight, bias)
        if self.hp.get("permute_features", True):
            perm = torch.randperm(H, device=dev)
            X = X[:, perm]
        X = self._add_onehot_feature_noise(X, scale)
        return X

    def _add_onehot_feature_noise(self, X: Tensor, scale: float) -> Tensor:
        """Per row, add N(0, scale^2) noise to a random subset of feature columns."""
        noise_frac = float(self.hp.get("x_onehot_noise_frac", 0.2))
        if noise_frac <= 0:
            return X
        T, H = X.shape
        if H == 0:
            return X
        n_noisy = min(H, max(1, int(round(noise_frac * H))))
        # Uniform random k-subset per row: first k entries of a random permutation.
        rand = torch.rand(T, H, device=X.device, dtype=X.dtype)
        idx = torch.argsort(rand, dim=1)[:, :n_noisy]
        rows = torch.arange(T, device=X.device).unsqueeze(1).expand(-1, n_noisy)
        X_out = X.clone()
        noise = torch.randn(T, n_noisy, device=X.device, dtype=X.dtype) * scale
        X_out[rows, idx] = X_out[rows, idx] + noise
        return X_out

    def _num2cat(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """Convert selected columns to categorical per the pre-computed plan.

        Returns
        -------
        X : Tensor
            Feature tensor with categorical conversions applied.
            One-hot columns expand the width; ordinal columns keep their width.
        continuous_mask : Tensor
            Boolean mask of shape ``(new_width,)`` where ``True`` marks
            continuous columns (eligible for outlier removal + scaling).
        """
        cat_plan: list[dict] = self.hp.get("cat_plan", [])
        n_cols = X.shape[1]

        if not cat_plan:
            return X, torch.ones(n_cols, dtype=torch.bool)

        n_cat = min(len(cat_plan), n_cols)
        perm = torch.randperm(n_cols)
        cat_col_indices = sorted(perm[:n_cat].tolist())
        cont_col_indices = sorted(perm[n_cat:].tolist())

        parts: list[Tensor] = []
        continuous_flags: list[bool] = []

        if cont_col_indices:
            parts.append(X[:, cont_col_indices])
            continuous_flags.extend([True] * len(cont_col_indices))

        for i in range(n_cat):
            col_data = X[:, cat_col_indices[i]]
            entry = cat_plan[i]
            nc = entry["num_cats"]
            is_onehot = entry["is_onehot"]

            assigner = MulticlassAssigner(
                nc,
                mode="rank",
                ordered_prob=0.0 if is_onehot else 1.0,
            )
            cat_data = assigner(col_data)

            if is_onehot:
                onehot = F.one_hot(cat_data.long(), nc).float()
                parts.append(onehot)
                continuous_flags.extend([False] * nc)
            else:
                if np.random.random() < float(
                    self.hp.get("cat_ordinal_permute_prob", 0.0)
                ):
                    cat_data = permute_classes(cat_data)
                parts.append(cat_data.unsqueeze(-1).float())
                continuous_flags.append(False)

        X_new = torch.cat(parts, dim=1)
        continuous_mask = torch.tensor(continuous_flags, dtype=torch.bool)
        return X_new, continuous_mask

    def _process_features(self, X: Tensor, continuous_mask: Tensor) -> Tensor:
        """Outlier removal and scaling on continuous columns, then column permutation.

        Parameters
        ----------
        X : Tensor
            Feature tensor of shape ``(T, H)``.
        continuous_mask : Tensor
            Boolean mask of shape ``(H,)``. ``True`` for continuous columns.
        """
        if continuous_mask.any():
            cont = X[:, continuous_mask].clone()
            cont = outlier_removing(cont, threshold=4)
            cont = standard_scaling(cont)
            X = X.clone()
            X[:, continuous_mask] = cont

        if self.hp.get("permute_features", True):
            perm = torch.randperm(X.shape[1], device=X.device)
            X = X[:, perm]

        return X
