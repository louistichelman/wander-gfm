"""Full-pool source-averaged Recall@K for CN / RA / PA."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from baselines.buddy.eval import _recall_ndcg_from_ranks
from baselines.nbfnet.data import NBFNetGraph
from data.eval_subsample import dataset_eval_subsample_seed

from .scores import HEURISTIC_METHODS, HeuristicAdj, score_sources, train_adjacency

DEFAULT_SOURCE_BATCH = 64


def _select_sources(
    targets: Dict[int, List[int]],
    *,
    max_queries: Optional[int],
    seed: int,
) -> List[int]:
    sources = sorted(s for s, tails in targets.items() if tails)
    if max_queries is not None and len(sources) > max_queries:
        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(len(sources), generator=g)[:max_queries]
        sources = [sources[int(i)] for i in perm.tolist()]
    return sources


def evaluate_heuristic_recall(
    graph: NBFNetGraph,
    method: str,
    *,
    split: str = "test",
    recall_k: int = 20,
    max_queries: Optional[int] = None,
    seed: int = 0,
    source_batch_size: int = DEFAULT_SOURCE_BATCH,
    adj: Optional[HeuristicAdj] = None,
) -> Dict[str, float]:
    """AnyGraph-style source-averaged Recall@K / NDCG@K on the full pool."""
    method = method.lower()
    if method not in HEURISTIC_METHODS:
        raise ValueError(f"unknown heuristic {method!r}; expected {HEURISTIC_METHODS}")
    if split == "val":
        targets = graph.val_targets or graph.test_targets
    else:
        targets = graph.test_targets
    ds_seed = dataset_eval_subsample_seed(seed, graph.bundle.name)
    sources = _select_sources(targets, max_queries=max_queries, seed=ds_seed)
    if not sources:
        return {f"recall@{recall_k}": 0.0, f"ndcg@{recall_k}": 0.0, "num_queries": 0.0}

    if adj is None:
        adj = train_adjacency(graph.edge_index, graph.num_nodes)

    train_out = graph.train_out_neighbors
    offset = graph.candidate_offset if graph.is_bipartite else 0
    num_nodes = graph.num_nodes
    recall_sum = 0.0
    ndcg_sum = 0.0
    source_count = 0
    batch = max(int(source_batch_size), 1)

    for start in range(0, len(sources), batch):
        batch_sources = sources[start : start + batch]
        scores = score_sources(adj, batch_sources, method)
        for i, src in enumerate(batch_sources):
            test_tails = targets.get(src, [])
            n_test = len(test_tails)
            if n_test == 0:
                continue
            row = torch.from_numpy(np.asarray(scores[i], dtype=np.float64)).clone()
            if graph.is_bipartite and offset > 0:
                row[:offset] = float("-inf")
            train_tails = train_out.get(src, [])
            if train_tails:
                filt = torch.tensor(sorted(set(int(t) for t in train_tails)), dtype=torch.long)
                valid = filt[(filt >= 0) & (filt < num_nodes)]
                if valid.numel() > 0:
                    row[valid] = float("-inf")
            if not graph.is_bipartite:
                row[src] = float("-inf")
            k = min(recall_k, int(row.numel()))
            top_locs = torch.topk(row, k).indices.tolist()
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


def evaluate_all_heuristics(
    graph: NBFNetGraph,
    *,
    methods: Sequence[str] = HEURISTIC_METHODS,
    recall_k: int = 20,
    max_queries: Optional[int] = None,
    seed: int = 0,
    source_batch_size: int = DEFAULT_SOURCE_BATCH,
    splits: Sequence[str] = ("val", "test"),
) -> Dict[str, Dict[str, float]]:
    """Score every requested method on ``splits``. Builds the train CSR once."""
    adj = train_adjacency(graph.edge_index, graph.num_nodes)
    out: Dict[str, Dict[str, float]] = {}
    for method in methods:
        metrics: Dict[str, float] = {
            "num_nodes": float(graph.num_nodes),
            "num_train_edges": float(graph.edge_index.shape[1]),
        }
        for split in splits:
            split_metrics = evaluate_heuristic_recall(
                graph,
                method,
                split=split,
                recall_k=recall_k,
                max_queries=max_queries,
                seed=seed,
                source_batch_size=source_batch_size,
                adj=adj,
            )
            for key, value in split_metrics.items():
                metrics[f"{split}/{key}"] = value
        out[method] = metrics
    return out
