"""Shared pairwise BCE for LP baselines (NBFNet, BUDDY, GCN)."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def pairwise_bce_loss(
    logits: Tensor, *, equal_pos_neg_weight: bool = False
) -> Tensor:
    """``logits`` is ``[B, 1+neg]`` with column 0 = positive.

    Default is a flat mean over all ``1+neg`` columns (positives are
    ``1/(1+neg)`` of the loss). ``equal_pos_neg_weight`` matches Wander's
    uniform reweight: the positive and the negative *set* each contribute
    half the row loss.
    """
    if logits.ndim != 2 or logits.shape[-1] < 1:
        raise ValueError(f"logits must be [B, 1+neg], got {tuple(logits.shape)}")
    target = torch.zeros_like(logits)
    target[:, 0] = 1.0
    if not equal_pos_neg_weight or logits.shape[-1] <= 1:
        return F.binary_cross_entropy_with_logits(logits, target)
    per = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    n_neg = logits.shape[-1] - 1
    weights = torch.ones_like(per)
    weights[:, 1:] = 1.0 / float(n_neg)
    return (per * weights).sum(dim=-1).div(weights.sum(dim=-1)).mean()
