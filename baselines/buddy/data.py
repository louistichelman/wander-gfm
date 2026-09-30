"""GraphBundle adapter and one-time hash / feature precomputation for BUDDY."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import Tensor

from baselines.nbfnet.data import NBFNetGraph, bundle_to_nbfnet_graph, iter_triple_batches

from .hashing import ElphHashes, HashConfig, HashTable, move_hash_tables
from .model import propagate_features

# Featured reporting-set folders (node features used when present).
FEATURED_ANYGRAPH = frozenset(
    {
        "ANYGRAPH_CITESEER",
        "ANYGRAPH_CORA",
        "ANYGRAPH_PUBMED",
        "ANYGRAPH_CS",
        "ANYGRAPH_PRODUCTS_HOME",
    }
)


@dataclass
class BuddyGraph:
    """NBFNet graph plus precomputed sketches and (optional) SIGN features."""

    graph: NBFNetGraph
    hasher: ElphHashes
    hash_tables: HashTable
    cards: Tensor
    degrees: Tensor
    x: Optional[Tensor]
    raw_feature_dim: Optional[int]
    use_feature: bool
    use_embedding: bool
    val_targets: Dict[int, List[int]]
    test_targets: Dict[int, List[int]]
    preferred_eval_split: str


def _degrees_from_edges(edge_index: Tensor, num_nodes: int) -> Tensor:
    deg = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
    if edge_index.numel() == 0:
        return deg
    ones = torch.ones(edge_index.size(1), dtype=torch.float32, device=edge_index.device)
    deg.scatter_add_(0, edge_index[0], ones)
    return deg


def _build_out_neighbors(triples: Tensor) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    if triples.numel() == 0:
        return out
    for row in triples.tolist():
        h, _r, t = int(row[0]), int(row[1]), int(row[2])
        out.setdefault(h, []).append(t)
    return out


def should_use_features(graph: NBFNetGraph, flag: bool) -> bool:
    if not flag:
        return False
    name = str(graph.bundle.name)
    if name in FEATURED_ANYGRAPH:
        return graph.x is not None
    return False


def prepare_buddy_graph(
    graph: NBFNetGraph,
    *,
    hash_cfg: HashConfig,
    use_node_features: bool,
    sign_k: int,
    device: torch.device,
) -> BuddyGraph:
    """Build hashes on train-visible edges and optional SIGN node features."""
    hasher = ElphHashes(hash_cfg)
    edge_index = graph.edge_index.to(device)
    tables, cards = hasher.build_hash_tables(graph.num_nodes, edge_index)
    degrees = _degrees_from_edges(edge_index, graph.num_nodes)

    use_feature = should_use_features(graph, use_node_features)
    raw_dim: Optional[int] = None
    x: Optional[Tensor] = None
    if use_feature and graph.x is not None:
        raw = graph.x.to(device=device, dtype=torch.float32)
        raw_dim = int(raw.shape[1])
        x = propagate_features(raw, edge_index, sign_k)

    preferred = str(graph.bundle.metadata.get("preferred_eval_split") or "val")
    val_targets = _build_out_neighbors(graph.val_triples)
    test_targets = graph.test_targets
    if not val_targets:
        preferred = "test"

    return BuddyGraph(
        graph=graph,
        hasher=hasher,
        hash_tables=tables,
        cards=cards,
        degrees=degrees,
        x=x,
        raw_feature_dim=raw_dim,
        use_feature=use_feature,
        use_embedding=not use_feature,
        val_targets=val_targets,
        test_targets=test_targets,
        preferred_eval_split=preferred,
    )


def buddy_to_device(state: BuddyGraph, device: torch.device) -> BuddyGraph:
    state.hash_tables = move_hash_tables(state.hash_tables, device)
    state.cards = state.cards.to(device)
    state.degrees = state.degrees.to(device)
    if state.x is not None:
        state.x = state.x.to(device)
    graph = state.graph
    graph.edge_index = graph.edge_index.to(device)
    graph.edge_type = graph.edge_type.to(device)
    graph.train_triples = graph.train_triples.to(device)
    graph.val_triples = graph.val_triples.to(device)
    graph.test_triples = graph.test_triples.to(device)
    return state


def sample_negative_dst(
    num_nodes: int,
    src: Tensor,
    num_negative: int,
    *,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Same-source negatives: keep ``src``, sample random destinations.

    Unlike official BUDDY's static global non-edge list, each positive
    ``(u, v)`` is paired with ``num_negative`` random ``(u, v')``.
    """
    bsz = src.shape[0]
    src_cpu = src.detach().cpu().unsqueeze(1)
    negs = torch.randint(0, num_nodes, (bsz, num_negative), generator=generator)
    for _ in range(8):
        bad = negs == src_cpu
        if not bool(bad.any()):
            break
        negs = torch.where(
            bad,
            torch.randint(0, num_nodes, (bsz, num_negative), generator=generator),
            negs,
        )
    return negs.to(src.device)


__all__ = [
    "BuddyGraph",
    "FEATURED_ANYGRAPH",
    "bundle_to_nbfnet_graph",
    "buddy_to_device",
    "iter_triple_batches",
    "prepare_buddy_graph",
    "sample_negative_dst",
    "should_use_features",
]
