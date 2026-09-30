"""Train-graph CN / RA / PA scores for homogeneous link prediction.

All three scores use only the visible train message-passing graph (no val/test
edges). Homogeneous AnyGraph graphs are stored undirected, so both directions
are already present in ``edge_index``.

CN(u,v) = |N(u) ∩ N(v)|
RA(u,v) = sum_{w in N(u) ∩ N(v)} 1 / d(w)
PA(u,v) = d(u) * d(v)

CN and RA have a large zero mass, so ranking by the raw score alone falls back
to node-id order via ``topk``. A PA tie-break is mixed in at a scale that cannot
change the primary ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor

HEURISTIC_METHODS: tuple[str, ...] = ("cn", "ra", "pa")

# Mixed into CN/RA so equal primary scores rank by PA instead of node id.
# CN is integer-valued; RA differences are at least ~1/d_max^2. This scale
# stays below both.
_TIEBREAK_SCALE = 1e-12


@dataclass
class HeuristicAdj:
    """Binary CSR train adjacency plus RA-scaled rows and degrees."""

    A: sp.csr_matrix
    A_ra: sp.csr_matrix
    degrees: np.ndarray  # [N] float64

    @property
    def num_nodes(self) -> int:
        return int(self.A.shape[0])


def train_adjacency(edge_index: Tensor, num_nodes: int) -> HeuristicAdj:
    """Binary CSR from undirected train ``edge_index`` (self-loops dropped)."""
    src = edge_index[0].detach().cpu().numpy().astype(np.int64, copy=False)
    dst = edge_index[1].detach().cpu().numpy().astype(np.int64, copy=False)
    mask = src != dst
    src = src[mask]
    dst = dst[mask]
    if src.size == 0:
        A = sp.csr_matrix((num_nodes, num_nodes), dtype=np.float64)
    else:
        data = np.ones(src.shape[0], dtype=np.float64)
        A = sp.csr_matrix((data, (src, dst)), shape=(num_nodes, num_nodes))
        A.data[:] = 1.0
        A.eliminate_zeros()
        A = (A > 0).astype(np.float64).tocsr()
    degrees = np.asarray(A.sum(axis=1), dtype=np.float64).ravel()
    inv = np.zeros_like(degrees)
    nz = degrees > 0
    inv[nz] = 1.0 / degrees[nz]
    A_ra = A.multiply(inv[:, np.newaxis]).tocsr()
    return HeuristicAdj(A=A, A_ra=A_ra, degrees=degrees)


def score_sources(
    adj: HeuristicAdj,
    sources: Sequence[int] | np.ndarray | Tensor,
    method: str,
) -> np.ndarray:
    """Return ``[B, N]`` float64 scores for ``sources``.

    CN/RA include a PA tie-break that does not change the primary order.
    """
    method = method.lower()
    if method not in HEURISTIC_METHODS:
        raise ValueError(f"unknown heuristic {method!r}; expected {HEURISTIC_METHODS}")
    src = np.asarray(sources, dtype=np.int64).reshape(-1)
    n = adj.num_nodes
    if src.size == 0:
        return np.zeros((0, n), dtype=np.float64)
    pa = adj.degrees[src][:, None] * adj.degrees[None, :]
    if method == "pa":
        return pa
    rows = adj.A[src]
    primary = rows @ (adj.A if method == "cn" else adj.A_ra)
    if sp.issparse(primary):
        primary = primary.toarray()
    primary = np.asarray(primary, dtype=np.float64)
    pa_norm = pa / (float(np.max(adj.degrees) ** 2) + 1.0)
    return primary + _TIEBREAK_SCALE * pa_norm


def pair_scores(
    adj: HeuristicAdj,
    src: Iterable[int],
    dst: Iterable[int],
    method: str,
) -> np.ndarray:
    """Score individual ``(src, dst)`` pairs. Returns ``[B]``."""
    src_arr = np.asarray(list(src), dtype=np.int64)
    dst_arr = np.asarray(list(dst), dtype=np.int64)
    if src_arr.shape != dst_arr.shape:
        raise ValueError("src and dst must have the same length")
    unique_src, inv = np.unique(src_arr, return_inverse=True)
    batch = score_sources(adj, unique_src, method)
    return batch[inv, dst_arr]
