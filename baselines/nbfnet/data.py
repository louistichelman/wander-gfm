"""GraphBundle adapters for NBFNet training / evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data

from data.graph_bundle import (
    GraphBundle,
    _forward_query_mask,
    resolve_split,
)


@dataclass
class NBFNetGraph:
    """Message-passing graph + query triples extracted from a GraphBundle."""

    edge_index: Tensor  # [2, E_vis] visible (train) edges
    edge_type: Tensor  # [E_vis]
    num_nodes: int
    num_relations: int
    x: Optional[Tensor]
    train_triples: Tensor  # [T, 3] (h, r, t) forward train queries
    val_triples: Tensor
    test_triples: Tensor
    undirected_lp: bool
    has_inverse_edges: bool
    inverse_edge_map: Optional[Tensor]  # map over full original edge list
    full_edge_index: Tensor  # original full edge_index (for inverse lookup)
    full_edge_type: Tensor
    visible_edge_indices: Tensor  # indices into full edge list that are visible
    preferred_link_pred_eval: str
    max_eval_samples: Optional[int]
    is_bipartite: bool
    candidate_offset: int
    # source -> list of test / val tails (for recall@k)
    test_targets: Dict[int, List[int]]
    val_targets: Dict[int, List[int]]
    train_out_neighbors: Dict[int, List[int]]
    bundle: GraphBundle


def _triples_from_forward_mask(data: Data, split: str) -> Tensor:
    mask = _forward_query_mask(data, split)  # type: ignore[arg-type]
    # For undirected LP both dirs are marked; keep both as queries (matches Wander).
    # For KG inverses, _forward_query_mask already truncates to forward half.
    if getattr(data, "has_inverse_edges", False):
        n_forward = data.edge_index.shape[1] // 2
        mask = mask.clone()
        if mask.numel() > n_forward:
            mask = mask[:n_forward]
        idx = mask.nonzero(as_tuple=True)[0]
        # indices refer to forward half
    else:
        idx = mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return torch.zeros(0, 3, dtype=torch.long)
    h = data.edge_index[0, idx]
    t = data.edge_index[1, idx]
    r = data.edge_type[idx]
    return torch.stack([h, r, t], dim=-1)


def _build_out_neighbors(triples: Tensor) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for row in triples.tolist():
        h, _r, t = int(row[0]), int(row[1]), int(row[2])
        out.setdefault(h, []).append(t)
    return out


def bundle_to_nbfnet_graph(bundle: GraphBundle) -> NBFNetGraph:
    """Extract visible message graph and split query triples from ``bundle``."""
    data = resolve_split(bundle, "train")
    visible = data.visible_mask
    vis_idx = visible.nonzero(as_tuple=True)[0]
    edge_index = data.edge_index[:, vis_idx].contiguous()
    edge_type = data.edge_type[vis_idx].contiguous()

    train_triples = _triples_from_forward_mask(data, "train")
    val_triples = _triples_from_forward_mask(data, "val")
    test_triples = _triples_from_forward_mask(data, "test")

    x = data.x if getattr(data, "x", None) is not None else None
    preferred = str(bundle.metadata.get("preferred_link_pred_eval") or "mrr")
    meta_cap = bundle.metadata.get("max_eval_samples")
    max_eval_samples = int(meta_cap) if meta_cap is not None else None
    # AnyGraph stores these on Data (not bundle.metadata); ``is_bipartite`` is a
    # PyG method name so the loader uses ``anygraph_bipartite``.
    is_bipartite = bool(getattr(data, "anygraph_bipartite", False))
    candidate_offset = int(getattr(data, "anygraph_candidate_offset", 0) or 0)

    train_out = getattr(data, "anygraph_train_out_neighbors", None)
    if not isinstance(train_out, dict):
        train_out = _build_out_neighbors(train_triples)

    return NBFNetGraph(
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=int(data.num_nodes),
        num_relations=int(data.num_relations),
        x=x,
        train_triples=train_triples,
        val_triples=val_triples,
        test_triples=test_triples,
        undirected_lp=bool(getattr(data, "undirected_lp", False)),
        has_inverse_edges=bool(getattr(data, "has_inverse_edges", False)),
        inverse_edge_map=getattr(data, "inverse_edge_map", None),
        full_edge_index=data.edge_index,
        full_edge_type=data.edge_type,
        visible_edge_indices=vis_idx,
        preferred_link_pred_eval=preferred,
        max_eval_samples=max_eval_samples,
        is_bipartite=is_bipartite,
        candidate_offset=candidate_offset,
        test_targets=_build_out_neighbors(test_triples),
        val_targets=_build_out_neighbors(val_triples),
        train_out_neighbors=train_out,
        bundle=bundle,
    )


def remove_easy_edges(
    graph: NBFNetGraph,
    h_index: Tensor,
    r_index: Tensor,
    t_index: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Drop batch positive edges (and undirected/inverse reverses) from the message graph.

    ``t_index`` is ``[B]`` positive tails (only positives need removing).
    """
    edge_index = graph.edge_index
    edge_type = graph.edge_type
    if h_index.numel() == 0:
        return edge_index, edge_type

    device = edge_index.device
    h = h_index.view(-1).to(device)
    r = r_index.view(-1).to(device)
    t = t_index.view(-1).to(device)

    src, dst = edge_index[0], edge_index[1]
    # Match any (h,r,t) in the batch via broadcasting chunks to limit memory
    keep = torch.ones(src.shape[0], dtype=torch.bool, device=device)
    chunk = 64
    for start in range(0, h.shape[0], chunk):
        hs = h[start : start + chunk].view(-1, 1)
        rs = r[start : start + chunk].view(-1, 1)
        ts = t[start : start + chunk].view(-1, 1)
        hit = (src.view(1, -1) == hs) & (dst.view(1, -1) == ts) & (edge_type.view(1, -1) == rs)
        keep &= ~hit.any(dim=0)
        if graph.undirected_lp or graph.has_inverse_edges:
            if graph.has_inverse_edges:
                inv_r = (rs + graph.num_relations // 2) % graph.num_relations
            else:
                inv_r = rs
            hit_rev = (
                (src.view(1, -1) == ts)
                & (dst.view(1, -1) == hs)
                & (edge_type.view(1, -1) == inv_r)
            )
            keep &= ~hit_rev.any(dim=0)

    return edge_index[:, keep], edge_type[keep]


def sample_negative_tails(
    num_nodes: int,
    positives: Tensor,
    num_negative: int,
    *,
    generator: Optional[torch.Generator] = None,
    exclude: Optional[Dict[Tuple[int, int], set]] = None,
) -> Tensor:
    """Sample ``num_negative`` random tails per positive triple.

    Returns ``[B, 1 + num_negative, 3]`` with column 0 = positive.
    """
    del exclude  # reserved for filtered negatives; unused for speed
    bsz = positives.shape[0]
    device = positives.device
    out = positives.new_zeros(bsz, 1 + num_negative, 3)
    out[:, 0] = positives
    # Vectorized: sample with replacement then fix collisions with the positive
    # (and rare duplicates) with a few resample passes. Generator is CPU-only.
    h = positives[:, 0:1].expand(-1, num_negative)
    r = positives[:, 1:2].expand(-1, num_negative)
    t_pos = positives[:, 2].detach().cpu().unsqueeze(1)
    negs = torch.randint(0, num_nodes, (bsz, num_negative), generator=generator)
    for _ in range(8):
        bad = negs == t_pos
        if not bool(bad.any()):
            break
        negs = torch.where(
            bad,
            torch.randint(0, num_nodes, (bsz, num_negative), generator=generator),
            negs,
        )
    out[:, 1:, 0] = h
    out[:, 1:, 1] = r
    out[:, 1:, 2] = negs.to(device)
    return out


def iter_triple_batches(
    triples: Tensor,
    batch_size: int,
    *,
    shuffle: bool = True,
    generator: Optional[torch.Generator] = None,
    max_batches: Optional[int] = None,
):
    """Yield batches of triples.

    If ``max_batches`` is set and ``len(triples) > max_batches * batch_size``,
    sample ``max_batches`` random batches with replacement (fixed steps/epoch).
    Otherwise shuffle once and iterate the full train set (normal epoch).
    """
    n = triples.shape[0]
    if n == 0:
        return
    budget = None if max_batches is None else int(max_batches) * int(batch_size)
    if budget is not None and n > budget:
        # Fixed number of optimizer steps: sample each batch independently.
        for _ in range(int(max_batches)):
            if generator is None:
                idx = torch.randint(0, n, (batch_size,))
            else:
                idx = torch.randint(0, n, (batch_size,), generator=generator)
            yield triples[idx]
        return
    if shuffle:
        perm = torch.randperm(n, generator=generator)
        triples = triples[perm]
    for start in range(0, n, batch_size):
        yield triples[start : start + batch_size]


def subsample_triples(
    triples: Tensor,
    max_queries: Optional[int],
    *,
    seed: int = 0,
) -> Tensor:
    if max_queries is None or triples.shape[0] <= max_queries:
        return triples
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(triples.shape[0], generator=g)[:max_queries]
    return triples[perm]
