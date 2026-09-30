"""Full-pool source-averaged Recall@K / NDCG@K for BUDDY pair scores."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor, nn

from data.eval_subsample import dataset_eval_subsample_seed

from .data import BuddyGraph
from .hashing import ElphHashes, HashTable
from .model import BUDDY


def score_pairs(
    model: BUDDY,
    hasher: ElphHashes,
    hash_tables: HashTable,
    cards: Tensor,
    src: Tensor,
    dst: Tensor,
    *,
    x: Optional[Tensor] = None,
    degrees: Optional[Tensor] = None,
    emb: Optional[Tensor] = None,
) -> Tensor:
    """Score ``(src, dst)`` pairs. Returns logits ``[B]``."""
    links = torch.stack([src, dst], dim=1)
    sf = hasher.get_subgraph_features(links, hash_tables, cards)
    node_features = None
    if model.use_feature and x is not None:
        node_features = x[links]
    src_deg = degrees[src] if degrees is not None and model.append_normalised else None
    dst_deg = degrees[dst] if degrees is not None and model.append_normalised else None
    batch_emb = emb[links] if emb is not None else None
    return model(sf, node_features, src_deg, dst_deg, batch_emb).view(-1)


def _recall_ndcg_from_ranks(
    test_tails: Sequence[int],
    top_locs: Sequence[int],
    recall_k: int,
) -> tuple[float, float]:
    n_test = len(test_tails)
    if n_test == 0:
        return 0.0, 0.0
    top_rank = {int(t): rk for rk, t in enumerate(top_locs)}
    hits = 0
    dcg = 0.0
    for t in test_tails:
        rk = top_rank.get(int(t))
        if rk is not None:
            hits += 1
            dcg += 1.0 / math.log2(rk + 2)
    max_dcg = sum(1.0 / math.log2(rk + 2) for rk in range(min(n_test, recall_k)))
    recall = hits / n_test
    ndcg = (dcg / max_dcg) if max_dcg > 0 else 0.0
    return recall, ndcg


@torch.no_grad()
def evaluate_recall_at_k(
    model: BUDDY,
    state: BuddyGraph,
    *,
    device: torch.device,
    embedding: Optional[nn.Embedding] = None,
    recall_k: int = 20,
    max_queries: Optional[int] = None,
    seed: int = 0,
    pair_batch_size: int = 250_000,
    split: str = "test",
) -> Dict[str, float]:
    """AnyGraph-style source-averaged Recall@K on the full candidate pool."""
    model.eval()
    graph = state.graph
    targets: Dict[int, List[int]] = state.test_targets if split == "test" else state.val_targets
    if split != "test" and not targets:
        targets = state.test_targets
        split = "test"
    sources = sorted(targets.keys())
    if max_queries is not None and len(sources) > max_queries:
        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(len(sources), generator=g)[:max_queries]
        sources = [sources[int(i)] for i in perm.tolist()]

    train_out = graph.train_out_neighbors
    offset = graph.candidate_offset if graph.is_bipartite else 0
    num_nodes = graph.num_nodes
    cand = torch.arange(offset, num_nodes, dtype=torch.long, device=device)
    if cand.numel() == 0:
        return {f"recall@{recall_k}": 0.0, f"ndcg@{recall_k}": 0.0, "num_queries": 0.0}

    emb_table: Optional[Tensor] = None
    if embedding is not None:
        if model.propagate_embeddings:
            emb_table = model.propagate_embeddings_func(embedding, graph.edge_index)
        else:
            emb_table = embedding.weight

    recall_sum = 0.0
    ndcg_sum = 0.0
    source_count = 0
    for src in sources:
        test_tails = targets.get(src, [])
        if not test_tails:
            continue
        scores = torch.empty(cand.numel(), dtype=torch.float32, device=device)
        src_t = torch.full((pair_batch_size,), src, dtype=torch.long, device=device)
        for start in range(0, cand.numel(), pair_batch_size):
            dst = cand[start : start + pair_batch_size]
            src_b = src_t[: dst.numel()]
            scores[start : start + dst.numel()] = score_pairs(
                model,
                state.hasher,
                state.hash_tables,
                state.cards,
                src_b,
                dst,
                x=state.x,
                degrees=state.degrees,
                emb=emb_table,
            )
        scores_cpu = scores.float().cpu()
        train_tails = train_out.get(src, [])
        if train_tails:
            local = [t - offset for t in train_tails if t >= offset]
            if local:
                scores_cpu[torch.tensor(local, dtype=torch.long)] = float("-inf")
        if not graph.is_bipartite and offset <= src < num_nodes:
            scores_cpu[src - offset] = float("-inf")

        k = min(recall_k, scores_cpu.numel())
        top_local = torch.topk(scores_cpu, k).indices.tolist()
        top_locs = [int(i) + offset for i in top_local]
        rec, ndcg = _recall_ndcg_from_ranks(test_tails, top_locs, recall_k)
        recall_sum += rec
        ndcg_sum += ndcg
        source_count += 1

    denom = max(source_count, 1)
    return {
        f"recall@{recall_k}": recall_sum / denom,
        f"ndcg@{recall_k}": ndcg_sum / denom,
        "num_queries": float(source_count),
    }


def evaluate_on_split(
    model: BUDDY,
    state: BuddyGraph,
    *,
    split: str,
    max_queries: Optional[int],
    seed: int,
    recall_k: int,
    device: torch.device,
    embedding: Optional[nn.Embedding] = None,
    pair_batch_size: int = 250_000,
) -> Dict[str, float]:
    ds_seed = dataset_eval_subsample_seed(seed, state.graph.bundle.name)
    return evaluate_recall_at_k(
        model,
        state,
        device=device,
        embedding=embedding,
        recall_k=recall_k,
        max_queries=max_queries,
        seed=ds_seed,
        pair_batch_size=pair_batch_size,
        split=split,
    )
