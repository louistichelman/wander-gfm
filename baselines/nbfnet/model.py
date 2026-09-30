"""NBFNet (Zhu et al.) for multi-relational link prediction.

Self-contained DistMult + sum-aggregate implementation. Optional node-feature
injection adds a linear projection of ``x`` into the Bellman-Ford boundary.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class RelationalConv(nn.Module):
    """One NBFNet layer: DistMult messages, sum aggregate, concat boundary, linear."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_relation: int,
        *,
        layer_norm: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.relation = nn.Embedding(num_relation, input_dim)
        self.linear = nn.Linear(input_dim * 2, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim) if layer_norm else None
        self.activation = getattr(F, activation) if isinstance(activation, str) else activation

    def forward(
        self,
        hidden: torch.Tensor,
        boundary: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            hidden / boundary: ``[B, N, D]``
            edge_index: ``[2, E]``
            edge_type: ``[E]``
        """
        src, dst = edge_index[0], edge_index[1]
        rel = self.relation(edge_type)  # [E, D]
        msg = hidden[:, src] * rel.unsqueeze(0)  # [B, E, D]
        bsz, num_nodes, dim = hidden.shape
        agg = torch.zeros(bsz, num_nodes, dim, device=hidden.device, dtype=hidden.dtype)
        index = dst.view(1, -1, 1).expand(bsz, -1, dim)
        agg.scatter_add_(1, index, msg)
        out = self.linear(torch.cat([agg, boundary], dim=-1))
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        if self.activation is not None:
            out = self.activation(out)
        return out


class NBFNet(nn.Module):
    """Query-conditioned path GNN for link prediction (tail ranking)."""

    def __init__(
        self,
        input_dim: int,
        num_relation: int,
        hidden_dims: Optional[Sequence[int]] = None,
        *,
        short_cut: bool = True,
        layer_norm: bool = True,
        activation: str = "relu",
        num_mlp_layer: int = 2,
        use_node_features: bool = False,
        node_feature_dim: Optional[int] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32] * 6
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = int(num_relation)
        self.short_cut = short_cut
        self.use_node_features = bool(use_node_features)

        self.query = nn.Embedding(self.num_relation, input_dim)
        self.layers = nn.ModuleList(
            [
                RelationalConv(
                    self.dims[i],
                    self.dims[i + 1],
                    self.num_relation,
                    layer_norm=layer_norm,
                    activation=activation,
                )
                for i in range(len(self.dims) - 1)
            ]
        )

        if self.use_node_features:
            if node_feature_dim is None:
                raise ValueError("node_feature_dim is required when use_node_features=True")
            self.feature_linear = nn.Linear(int(node_feature_dim), input_dim)
        else:
            self.feature_linear = None

        feature_dim = self.dims[-1] + input_dim
        mlp: list[nn.Module] = []
        for _ in range(num_mlp_layer - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)

    def bellmanford(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        num_nodes: int,
        h_index: torch.Tensor,
        r_index: torch.Tensor,
        node_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run L message-passing steps; return ``[B, N, D_out + D_in]`` node states."""
        batch_size = h_index.shape[0]
        device = h_index.device
        query = self.query(r_index)  # [B, D]

        boundary = torch.zeros(batch_size, num_nodes, self.dims[0], device=device, dtype=query.dtype)
        if self.feature_linear is not None and node_features is not None:
            feat = self.feature_linear(node_features)  # [N, D]
            boundary = boundary + feat.unsqueeze(0)
        boundary.scatter_add_(
            1, h_index.view(-1, 1, 1).expand(-1, 1, self.dims[0]), query.unsqueeze(1)
        )

        hidden = boundary
        for layer in self.layers:
            update = layer(hidden, boundary, edge_index, edge_type)
            if self.short_cut and update.shape == hidden.shape:
                update = update + hidden
            hidden = update

        node_query = query.unsqueeze(1).expand(-1, num_nodes, -1)
        return torch.cat([hidden, node_query], dim=-1)

    def score_tails(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        num_nodes: int,
        h_index: torch.Tensor,
        r_index: torch.Tensor,
        t_index: Optional[torch.Tensor] = None,
        node_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Score candidate tails.

        If ``t_index`` is None, return scores for all nodes ``[B, N]``.
        Else ``t_index`` is ``[B, K]`` and returns ``[B, K]``.
        """
        features = self.bellmanford(
            edge_index, edge_type, num_nodes, h_index, r_index, node_features=node_features
        )
        if t_index is None:
            return self.mlp(features).squeeze(-1)
        gather_index = t_index.unsqueeze(-1).expand(-1, -1, features.shape[-1])
        selected = features.gather(1, gather_index)
        return self.mlp(selected).squeeze(-1)

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        num_nodes: int,
        batch: torch.Tensor,
        node_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Training forward.

        ``batch`` shape ``[B, 1 + num_neg, 3]`` with columns ``(h, r, t)``. All
        rows share the same ``(h, r)``; column 0 is the positive tail.
        """
        h_index = batch[:, 0, 0]
        r_index = batch[:, 0, 1]
        t_index = batch[:, :, 2]
        return self.score_tails(
            edge_index,
            edge_type,
            num_nodes,
            h_index,
            r_index,
            t_index=t_index,
            node_features=node_features,
        )
