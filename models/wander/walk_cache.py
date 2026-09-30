"""Cached CSR / transition-probability structures for Wander random walks."""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import graph_walker
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data

if TYPE_CHECKING:
    from .wander import Wander

CSRTriple = tuple[np.ndarray, np.ndarray, np.ndarray]
_CACHE_ATTR = "_wander_walk_cache"


@dataclass
class RWWalkProfile:
    rw_indptr: np.ndarray
    rw_indices: np.ndarray
    rw_probs: np.ndarray
    neighbor_indptr: Optional[np.ndarray] = None
    neighbor_indices: Optional[np.ndarray] = None


@dataclass
class WalkGraphCache:
    num_nodes: int
    typed_forward: CSRTriple
    typed_transposed: CSRTriple
    rw_profiles: dict = field(default_factory=dict)


def _tensor_crc(t: Tensor) -> int:
    return zlib.crc32(t.detach().contiguous().cpu().numpy().tobytes())


def wander_cache_config_key(data: Data, wander: Wander) -> str:
    edge_index = data.edge_index
    edge_type = data.edge_type
    assert isinstance(edge_index, Tensor)
    assert isinstance(edge_type, Tensor)
    return repr(
        (
            tuple(edge_index.shape),
            tuple(edge_type.shape),
            _tensor_crc(edge_index),
            _tensor_crc(edge_type),
            int(data.num_nodes),
            int(data.num_relations),
            bool(wander.add_inverse_edges_kgs),
            bool(wander.only_forward_edges_when_inverses_added),
            bool(wander.record_neighbors),
        )
    )


def build_symmetrized_edge_index(edge_index: Tensor, remove_loops: bool) -> Tensor:
    edge_index_ = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
    edge_index_ = torch.unique(edge_index_, dim=1)
    if remove_loops:
        edge_index_ = edge_index_[:, edge_index_[0] != edge_index_[1]]
    return edge_index_


def _build_typed_csrs(data: Data, wander: Wander) -> tuple[CSRTriple, CSRTriple]:
    from .wander import to_csr_tensor

    edge_index = data.edge_index
    num_nodes = int(data.num_nodes)
    assert isinstance(edge_index, Tensor)

    forward_csr = to_csr_tensor(edge_index, data.edge_type, num_nodes)
    if wander.add_inverse_edges_kgs and wander.only_forward_edges_when_inverses_added:
        empty_edges = edge_index.new_zeros((2, 0))
        empty_types = data.edge_type.new_zeros((0,))
        transposed_csr = to_csr_tensor(empty_edges, empty_types, num_nodes)
    else:
        transposed_csr = to_csr_tensor(edge_index[[1, 0]], data.edge_type, num_nodes)
    return forward_csr, transposed_csr


def _build_rw_profile(
    edge_index: Tensor,
    num_nodes: int,
    remove_loops: bool,
    record_neighbors: bool,
) -> RWWalkProfile:
    from .wander import to_csr_tensor

    edge_index_ = build_symmetrized_edge_index(edge_index, remove_loops)
    rw_indptr, rw_indices, rw_probs = graph_walker.transition_probs_fast(
        Data(edge_index=edge_index_, num_nodes=num_nodes, is_directed_hash=False),
    )

    neighbor_indptr = None
    neighbor_indices = None
    if record_neighbors:
        neighbor_indptr, neighbor_indices, _ = to_csr_tensor(
            edge_index_, torch.ones_like(edge_index_[0]), num_nodes
        )

    return RWWalkProfile(
        rw_indptr=rw_indptr,
        rw_indices=rw_indices,
        rw_probs=rw_probs,
        neighbor_indptr=neighbor_indptr,
        neighbor_indices=neighbor_indices,
    )


def _build_walk_graph_cache(data: Data, wander: Wander) -> WalkGraphCache:
    typed_forward, typed_transposed = _build_typed_csrs(data, wander)
    return WalkGraphCache(
        num_nodes=int(data.num_nodes),
        typed_forward=typed_forward,
        typed_transposed=typed_transposed,
    )


def get_walk_graph_cache(data: Data, wander: Wander) -> WalkGraphCache:
    key = wander_cache_config_key(data, wander)
    caches: dict[str, WalkGraphCache] = getattr(data, _CACHE_ATTR, None)
    if caches is None:
        caches = {}
        setattr(data, _CACHE_ATTR, caches)
    if key not in caches:
        caches[key] = _build_walk_graph_cache(data, wander)
    return caches[key]


def get_rw_profile(
    cache: WalkGraphCache,
    data: Data,
    wander: Wander,
    remove_loops: bool,
) -> RWWalkProfile:
    """Return (and cache) an RW transition profile for ``data.edge_index``."""
    key = remove_loops
    if key not in cache.rw_profiles:
        assert isinstance(data.edge_index, Tensor)
        cache.rw_profiles[key] = _build_rw_profile(
            data.edge_index,
            cache.num_nodes,
            remove_loops,
            wander.record_neighbors,
        )
    return cache.rw_profiles[key]


def clear_walk_graph_cache(data: Data) -> None:
    if hasattr(data, _CACHE_ATTR):
        delattr(data, _CACHE_ATTR)
