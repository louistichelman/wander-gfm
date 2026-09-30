"""Shared SCM building blocks: input sampling, Gaussian noise, and SemiGraphConv."""

from __future__ import annotations

import random
from functools import partial
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch_geometric.utils import add_self_loops, degree, softmax


class GaussianNoise(nn.Module):
    def __init__(self, std):
        super().__init__()
        self.std = std

    def forward(self, X):
        return X + torch.normal(torch.zeros_like(X), self.std)


class XSampler:
    """Input sampler for generating features for prior datasets.

    Supports multiple feature distribution types:
    - Normal: Standard normal distribution
    - Multinomial: Categorical features with random number of categories
    - Zipf: Power law distributed features
    - Mixed: Random combination of the above

    Parameters
    ----------
    seq_len : int
        Length of sequence to generate

    num_features : int
        Number of features to generate

    pre_stats : bool
        Whether to pre-generate statistics for the input features

    sampling : str, default='mixed'
        Feature sampling strategy ('normal', 'mixed', 'uniform')

    device : str, default='cpu'
        Device to store tensors on
    """

    def __init__(
        self, seq_len, num_features, pre_stats=False, sampling="mixed", device="cpu"
    ):
        self.seq_len = seq_len
        self.num_features = num_features
        self.pre_stats = pre_stats
        self.sampling = sampling
        self.device = device

        if pre_stats:
            self._pre_stats()

    def _pre_stats(self):
        means = np.random.normal(0, 1, self.num_features)
        stds = np.abs(np.random.normal(0, 1, self.num_features) * means)
        self.means = (
            torch.tensor(means, dtype=torch.float, device=self.device)
            .unsqueeze(0)
            .repeat(self.seq_len, 1)
        )
        self.stds = (
            torch.tensor(stds, dtype=torch.float, device=self.device)
            .unsqueeze(0)
            .repeat(self.seq_len, 1)
        )

    def sample(self, return_numpy=False):
        """Generate features according to the specified sampling strategy.

        Returns
        -------
        X : torch.Tensor
            Generated features of shape (seq_len, num_features)
        """
        samplers = {
            "normal": self.sample_normal_all,
            "mixed": self.sample_mixed,
            "uniform": self.sample_uniform,
        }
        if self.sampling not in samplers:
            raise ValueError(f"Invalid sampling method: {self.sampling}")
        X = samplers[self.sampling]()

        return X.cpu().numpy() if return_numpy else X

    def sample_normal_all(self):
        if self.pre_stats:
            X = torch.normal(self.means, self.stds.abs()).float()
        else:
            X = torch.normal(
                0.0, 1.0, (self.seq_len, self.num_features), device=self.device
            ).float()
        return X

    def sample_uniform(self):
        """Generate uniformly distributed features."""
        return torch.rand((self.seq_len, self.num_features), device=self.device)

    def sample_normal(self, n=None):
        """Generate normally distributed features.

        Parameters
        ----------
        n : int
            Index of the feature to generate
        """

        if self.pre_stats:
            return torch.normal(self.means[:, n], self.stds[:, n].abs()).float()
        else:
            return torch.normal(0.0, 1.0, (self.seq_len,), device=self.device).float()

    def sample_multinomial(self):
        """Generate categorical features."""
        n_categories = random.randint(2, 20)
        probs = torch.rand(n_categories, device=self.device)
        x = torch.multinomial(probs, self.seq_len, replacement=True)
        x = x.float()
        return (x - x.mean()) / x.std()

    def sample_zipf(self):
        """Generate Zipf-distributed features."""
        x = np.random.zipf(2.0 + random.random() * 2, (self.seq_len,))
        x = torch.tensor(x, device=self.device).clamp(max=10)
        x = x.float()
        return x - x.mean()

    def sample_mixed(self):
        """Generate features with mixed distributions."""
        X = []
        zipf_p, multi_p, normal_p = (
            random.random() * 0.66,
            random.random() * 0.66,
            random.random() * 0.66,
        )
        for n in range(self.num_features):
            if random.random() > normal_p:
                x = self.sample_normal(n)
            elif random.random() > multi_p:
                x = self.sample_multinomial()
            elif random.random() > zipf_p:
                x = self.sample_zipf()
            else:
                x = torch.rand((self.seq_len,), device=self.device)
            X.append(x)
        return torch.stack(X, -1)


class GCNConv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, edge_index: Tensor, x: Tensor, num_nodes: int) -> Tensor:
        src, dst = edge_index[0], edge_index[1]
        deg = degree(src, num_nodes=num_nodes).float().clamp(min=1.0)
        norm = 1.0 / (deg[src] * deg[dst]).sqrt()
        messages = x[src] * norm.unsqueeze(-1)
        out = torch.zeros_like(x)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)
        return out


class SAGEConv(nn.Module):
    def __init__(self, mode: Literal["mean", "min", "max"] = "mean"):
        super().__init__()
        self.mode = mode

    def forward(self, edge_index: Tensor, x: Tensor, num_nodes: int) -> Tensor:
        src, dst = edge_index[0], edge_index[1]
        if self.mode == "mean":
            out = torch.zeros_like(x)
            count = torch.zeros(num_nodes, 1, device=x.device)
            out.scatter_add_(0, dst.unsqueeze(-1).expand_as(x[src]), x[src])
            count.scatter_add_(0, dst.unsqueeze(-1), torch.ones(src.size(0), 1, device=x.device))
            count = count.clamp(min=1.0)
            return out / count
        elif self.mode == "min":
            out = torch.full_like(x, float("inf"))
            out.scatter_reduce_(0, dst.unsqueeze(-1).expand_as(x[src]), x[src], reduce="amin", include_self=False)
            out[out == float("inf")] = 0.0
            return out
        elif self.mode == "max":
            out = torch.full_like(x, float("-inf"))
            out.scatter_reduce_(0, dst.unsqueeze(-1).expand_as(x[src]), x[src], reduce="amax", include_self=False)
            out[out == float("-inf")] = 0.0
            return out
        else:
            raise ValueError(f"Unknown SAGEConv mode: {self.mode}")


class GTConv(nn.Module):
    def __init__(self, d, n_heads=1):
        super().__init__()
        assert d % n_heads == 0

        self.d = d
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.attn_scores_coef = 1.0 / self.d_head**0.5

        self.attn_qkv_linear = nn.Linear(d, d * 3)
        self.output_linear = nn.Linear(d, d)

    def forward(self, edge_index: Tensor, x: Tensor, num_nodes: int) -> Tensor:
        assert x.ndim == 2, "Batching is not supported yet"
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        src, dst = edge_index[0], edge_index[1]

        qkv = self.attn_qkv_linear(x)
        qkv = qkv.reshape(-1, self.n_heads, self.d_head * 3)
        q, k, v = qkv.split(split_size=(self.d_head, self.d_head, self.d_head), dim=-1)

        attn_scores = (k[src] * q[dst]).sum(dim=-1, keepdim=True) * self.attn_scores_coef
        attn_probs = softmax(attn_scores, index=dst, num_nodes=num_nodes)

        messages = v[src] * attn_probs
        out = torch.zeros(num_nodes, self.n_heads, self.d_head, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(-1).unsqueeze(-1).expand_as(messages), messages)
        x = out.reshape(-1, self.d)

        x = self.output_linear(x)
        return x


# TODO: better name
class SemiGraphConv(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_output: int,
        conv_type: Literal[
            "gcn",
            "sage-mean",
            "sage-min",
            "sage-max",
            "gt",
        ],
        graph_conv_ratio: float,
    ):
        super().__init__()
        self.edge_index: Tensor | None = None
        self.num_nodes: int | None = None
        self.linear = nn.Linear(d_input, d_output)
        self.conv = {
            "gcn": GCNConv,
            "sage-mean": partial(SAGEConv, "mean"),
            "sage-min": partial(SAGEConv, "min"),
            "sage-max": partial(SAGEConv, "max"),
            "gt": partial(GTConv, d=d_output),
        }[conv_type]()
        self.mask = torch.bernoulli(
            graph_conv_ratio * torch.ones([d_output], dtype=torch.float32)
        ).bool()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.edge_index is not None and self.num_nodes is not None
        x = self.linear(x)
        assert self.edge_index.device == x.device
        message = self.conv(self.edge_index, x, self.num_nodes)
        y = torch.where(self.mask.to(x.device), message, x)
        return y
