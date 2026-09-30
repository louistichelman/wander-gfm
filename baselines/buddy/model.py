"""BUDDY MLP + SIGN-style feature / embedding propagation.

Port of ``subgraph-sketching`` BUDDY without ``torch_sparse``. GCN-normalized
propagation uses ``torch.sparse.mm``.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm


def build_gcn_adj(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Symmetric GCN-normalized sparse adjacency (self-loops included)."""
    ei, ew = gcn_norm(edge_index, num_nodes=num_nodes)
    return torch.sparse_coo_tensor(ei, ew, (num_nodes, num_nodes), device=edge_index.device).coalesce()


def propagate_features(x: Tensor, edge_index: Tensor, sign_k: int) -> Tensor:
    """Precompute SIGN polynomials.

    ``sign_k == 0`` applies one GCN-normalized hop (matches official BUDDY).
    ``sign_k > 0`` concatenates ``[x, Ax, ..., A^k x]``.
    """
    adj = build_gcn_adj(edge_index, x.size(0))
    if sign_k <= 0:
        return torch.sparse.mm(adj, x)
    xs = [x]
    cur = x
    for _ in range(sign_k):
        cur = torch.sparse.mm(adj, cur)
        xs.append(cur)
    return torch.cat(xs, dim=-1)


class SIGN(nn.Module):
    """Independent linear maps on each SIGN hop, then a joint projection."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, k: int, dropout: float):
        super().__init__()
        self.k = int(k)
        self.lins = nn.ModuleList(nn.Linear(in_channels, hidden_channels) for _ in range(self.k + 1))
        self.bns = nn.ModuleList(nn.BatchNorm1d(hidden_channels) for _ in range(self.k + 1))
        self.lin_out = nn.Linear((self.k + 1) * hidden_channels, out_channels)
        self.dropout = float(dropout)

    def forward(self, xs: Tensor) -> Tensor:
        chunks = torch.tensor_split(xs, self.k + 1, dim=-1)
        hs = []
        for lin, bn, chunk in zip(self.lins, self.bns, chunks):
            h = lin(chunk)
            h = torch.stack([bn(h[:, 0, :]), bn(h[:, 1, :])], dim=1)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            hs.append(h)
        return self.lin_out(torch.cat(hs, dim=-1))


class SIGNEmbedding(nn.Module):
    """SIGN applied to a full-graph embedding table."""

    def __init__(self, channels: int, k: int, dropout: float):
        super().__init__()
        self.k = int(k)
        self.lins = nn.ModuleList(nn.Linear(channels, channels) for _ in range(self.k + 1))
        self.bns = nn.ModuleList(nn.BatchNorm1d(channels) for _ in range(self.k + 1))
        self.lin_out = nn.Linear((self.k + 1) * channels, channels)
        self.dropout = float(dropout)
        self._adj: Optional[Tensor] = None

    def forward(self, x: Tensor, edge_index: Tensor, num_nodes: int) -> Tensor:
        if self._adj is None or self._adj.size(0) != num_nodes or self._adj.device != x.device:
            self._adj = build_gcn_adj(edge_index, num_nodes)
        adj = self._adj
        hs = []
        cur = x
        for lin, bn in zip(self.lins, self.bns):
            h = F.relu(bn(lin(cur)))
            h = F.dropout(h, p=self.dropout, training=self.training)
            hs.append(h)
            cur = torch.sparse.mm(adj, cur)
        return self.lin_out(torch.cat(hs, dim=-1))


class BUDDY(nn.Module):
    """Edge-wise MLP over subgraph sketches (+ optional features / embeddings)."""

    def __init__(
        self,
        *,
        num_struct_features: int,
        hidden_channels: int,
        num_features: Optional[int] = None,
        use_feature: bool = False,
        sign_k: int = 0,
        sign_dropout: float = 0.5,
        label_dropout: float = 0.5,
        feature_dropout: float = 0.5,
        add_normed_features: bool = False,
        use_embedding: bool = False,
        propagate_embeddings: bool = False,
    ):
        super().__init__()
        self.use_feature = bool(use_feature)
        self.use_embedding = bool(use_embedding)
        self.propagate_embeddings = bool(propagate_embeddings)
        self.append_normalised = bool(add_normed_features)
        self.sign_k = int(sign_k)
        self.label_dropout = float(label_dropout)
        self.feature_dropout = float(feature_dropout)
        self.dim = num_struct_features * 2 if self.append_normalised else num_struct_features

        if self.use_feature and self.sign_k != 0:
            if num_features is None:
                raise ValueError("num_features is required when use_feature and sign_k > 0")
            self.sign = SIGN(num_features, hidden_channels, hidden_channels, self.sign_k, sign_dropout)
        else:
            self.sign = None
        if self.use_embedding and self.propagate_embeddings and self.sign_k != 0:
            self.sign_embedding = SIGNEmbedding(hidden_channels, self.sign_k, sign_dropout)
        else:
            self.sign_embedding = None

        self.label_lin_layer = nn.Linear(self.dim, self.dim)
        self.bn_labels = nn.BatchNorm1d(self.dim)
        if self.use_feature:
            in_feat = num_features if num_features is not None else hidden_channels
            self.lin_feat = nn.Linear(in_feat, hidden_channels)
            self.lin_out = nn.Linear(hidden_channels, hidden_channels)
            self.bn_feats = nn.BatchNorm1d(hidden_channels)
        hidden = self.dim + (hidden_channels if self.use_feature else 0)
        if self.use_embedding:
            self.lin_emb = nn.Linear(hidden_channels, hidden_channels)
            self.lin_emb_out = nn.Linear(hidden_channels, hidden_channels)
            self.bn_embs = nn.BatchNorm1d(hidden_channels)
            hidden += hidden_channels
        self.lin = nn.Linear(hidden, 1)

    def _append_degree_normalised(self, x: Tensor, src_degree: Tensor, dst_degree: Tensor) -> Tensor:
        normaliser = torch.sqrt(src_degree * dst_degree).unsqueeze(1)
        normed = x / normaliser.clamp(min=1e-12)
        normed = torch.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.cat([x, normed], dim=1)

    def feature_forward(self, x: Tensor) -> Tensor:
        if self.sign is not None:
            x = self.sign(x)
        else:
            x = self.lin_feat(x)
        x = x[:, 0, :] * x[:, 1, :]
        x = self.lin_out(x)
        x = self.bn_feats(x)
        x = F.relu(x)
        return F.dropout(x, p=self.feature_dropout, training=self.training)

    def embedding_forward(self, x: Tensor) -> Tensor:
        x = self.lin_emb(x)
        x = x[:, 0, :] * x[:, 1, :]
        x = self.lin_emb_out(x)
        x = self.bn_embs(x)
        x = F.relu(x)
        return F.dropout(x, p=self.feature_dropout, training=self.training)

    def propagate_embeddings_func(self, embedding: nn.Embedding, edge_index: Tensor) -> Tensor:
        if self.sign_embedding is None:
            return embedding.weight
        return self.sign_embedding(embedding.weight, edge_index, embedding.num_embeddings)

    def forward(
        self,
        sf: Tensor,
        node_features: Optional[Tensor] = None,
        src_degree: Optional[Tensor] = None,
        dst_degree: Optional[Tensor] = None,
        emb: Optional[Tensor] = None,
    ) -> Tensor:
        if self.append_normalised:
            if src_degree is None or dst_degree is None:
                raise ValueError("degrees are required when add_normed_features is set")
            sf = self._append_degree_normalised(sf, src_degree, dst_degree)
        x = F.relu(self.bn_labels(self.label_lin_layer(sf)))
        x = F.dropout(x, p=self.label_dropout, training=self.training)
        if self.use_feature:
            if node_features is None:
                raise ValueError("node_features required when use_feature is set")
            x = torch.cat([x, self.feature_forward(node_features)], dim=1)
        if self.use_embedding:
            if emb is None:
                raise ValueError("emb required when use_embedding is set")
            x = torch.cat([x, self.embedding_forward(emb)], dim=1)
        return self.lin(x)
