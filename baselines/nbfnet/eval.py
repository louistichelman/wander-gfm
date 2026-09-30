"""Filtered MRR and Recall@K evaluation for NBFNet."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
from torch import Tensor, nn

from data.graph_bundle import (
    apply_filtered_entity_mask_,
    get_eval_filter_index,
    get_train_filter_index,
)
from data.lp_rank_stats import RankAccum, relation_names_from_bundle

from .data import NBFNetGraph, subsample_triples
from data.eval_subsample import dataset_eval_subsample_seed


@torch.no_grad()
def evaluate_filtered_mrr(
    model: nn.Module,
    graph: NBFNetGraph,
    triples: Tensor,
    *,
    device: torch.device,
    node_features: Optional[Tensor] = None,
    batch_size: int = 4,
) -> Dict[str, float]:
    """Filtered full-graph MRR / Hits@1/3/10 over ``triples`` ``[T, 3]``."""
    model.eval()
    filter_index = get_eval_filter_index(graph.bundle, "test")
    edge_index = graph.edge_index
    edge_type = graph.edge_type
    num_nodes = graph.num_nodes

    accum = RankAccum()

    for start in range(0, triples.shape[0], batch_size):
        batch = triples[start : start + batch_size]
        h = batch[:, 0].to(device)
        r = batch[:, 1].to(device)
        t = batch[:, 2]
        scores = model.score_tails(
            edge_index,
            edge_type,
            num_nodes,
            h,
            r,
            t_index=None,
            node_features=node_features,
        )  # [B, N]
        scores_cpu = scores.float().cpu()
        del scores

        for i in range(scores_cpu.shape[0]):
            hi, ri, ti = int(h[i]), int(r[i]), int(t[i])
            scores_i = scores_cpu[i].clone()
            filt = filter_index.tails.get((hi, ri))
            true_score = apply_filtered_entity_mask_(scores_i, filt, ti)
            # rank: number of scores strictly greater + 1 (ties against)
            rank = int(1 + (scores_i > true_score).sum().item())
            accum.add(rank, rel_id=ri)

    return accum.as_metrics(relation_names_from_bundle(graph.bundle))


@torch.no_grad()
def evaluate_recall_at_k(
    model: nn.Module,
    graph: NBFNetGraph,
    *,
    device: torch.device,
    node_features: Optional[Tensor] = None,
    recall_k: int = 20,
    max_queries: Optional[int] = None,
    seed: int = 0,
    batch_size: int = 4,
    split: str = "test",
) -> Dict[str, float]:
    """AnyGraph-style source-averaged Recall@K / NDCG@K."""
    model.eval()
    if split == "val":
        targets = graph.val_targets or graph.test_targets
    else:
        targets = graph.test_targets
    sources = sorted(targets.keys())
    if max_queries is not None and len(sources) > max_queries:
        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(len(sources), generator=g)[:max_queries]
        sources = [sources[int(i)] for i in perm.tolist()]

    # Prefer train-out-neighbors from triples; fall back to train filter index
    train_out = graph.train_out_neighbors
    filter_index = None
    if not train_out:
        filter_index = get_train_filter_index(graph.bundle, "test")

    recall_sum = 0.0
    ndcg_sum = 0.0
    source_count = 0

    for start in range(0, len(sources), batch_size):
        batch_sources = sources[start : start + batch_size]
        h = torch.tensor(batch_sources, dtype=torch.long, device=device)
        # AnyGraph / homogeneous LP uses relation 0 for forward edges; for
        # multi-rel recall we still score with r=0 only when num_relations==1,
        # otherwise use the dominant relation of each source's test edges.
        if graph.num_relations == 1 or (
            graph.num_relations == 2 and graph.has_inverse_edges
        ):
            r = torch.zeros_like(h)
        else:
            # Multiplex recall is not the preferred protocol; use r from first test edge
            r_list = []
            for src in batch_sources:
                # find a test triple for this source
                found = 0
                for row in graph.test_triples.tolist():
                    if int(row[0]) == src:
                        found = int(row[1])
                        break
                r_list.append(found)
            r = torch.tensor(r_list, dtype=torch.long, device=device)

        scores = model.score_tails(
            graph.edge_index,
            graph.edge_type,
            graph.num_nodes,
            h,
            r,
            t_index=None,
            node_features=node_features,
        )
        scores_cpu = scores.float().cpu()
        del scores

        for i, src in enumerate(batch_sources):
            test_tails = targets.get(src, [])
            n_test = len(test_tails)
            if n_test == 0:
                continue
            scores_i = scores_cpu[i].clone()
            train_tails = train_out.get(src, [])
            if train_tails:
                filt = torch.tensor(sorted(set(train_tails)), dtype=torch.long)
                scores_i[filt] = float("-inf")
            elif filter_index is not None:
                filt = filter_index.tails.get((src, int(r[i])))
                if filt is not None and filt.numel() > 0:
                    scores_i[filt] = float("-inf")
            if graph.is_bipartite and graph.candidate_offset > 0:
                scores_i[: graph.candidate_offset] = float("-inf")

            k = min(recall_k, scores_i.shape[0])
            top_locs = torch.topk(scores_i, k).indices.tolist()
            top_rank = {t: rk for rk, t in enumerate(top_locs)}

            hits = 0
            dcg = 0.0
            for t in test_tails:
                rk = top_rank.get(t)
                if rk is not None:
                    hits += 1
                    dcg += 1.0 / math.log2(rk + 2)
            max_dcg = sum(1.0 / math.log2(rk + 2) for rk in range(min(n_test, recall_k)))
            recall_sum += hits / n_test
            ndcg_sum += (dcg / max_dcg) if max_dcg > 0 else 0.0
            source_count += 1

    denom = max(source_count, 1)
    return {
        f"recall@{recall_k}": recall_sum / denom,
        f"ndcg@{recall_k}": ndcg_sum / denom,
        "num_queries": float(source_count),
    }


@torch.no_grad()
def evaluate_on_split(
    model: nn.Module,
    graph: NBFNetGraph,
    *,
    split: str,
    max_queries: Optional[int],
    seed: int,
    recall_k: int,
    device: torch.device,
    node_features: Optional[Tensor] = None,
    batch_size: int = 4,
) -> Dict[str, float]:
    """Dispatch to MRR or Recall@K based on ``graph.preferred_link_pred_eval``."""
    ds_seed = dataset_eval_subsample_seed(seed, graph.bundle.name)
    protocol = graph.preferred_link_pred_eval
    if protocol == "recall":
        return evaluate_recall_at_k(
            model,
            graph,
            device=device,
            node_features=node_features,
            recall_k=recall_k,
            max_queries=max_queries,
            seed=ds_seed,
            batch_size=batch_size,
            split=split,
        )

    triples = graph.val_triples if split == "val" else graph.test_triples
    if split == "train":
        triples = graph.train_triples
    triples = subsample_triples(triples, max_queries, seed=ds_seed)
    return evaluate_filtered_mrr(
        model,
        graph,
        triples,
        device=device,
        node_features=node_features,
        batch_size=batch_size,
    )


def evaluate_nbfnet(
    model: nn.Module,
    graph: NBFNetGraph,
    *,
    max_queries: Optional[int] = None,
    seed: int = 0,
    recall_k: int = 20,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Public helper: evaluate on the test split."""
    if device is None:
        device = next(model.parameters()).device
    x = graph.x.to(device) if model.use_node_features and graph.x is not None else None
    return evaluate_on_split(
        model,
        graph,
        split="test",
        max_queries=max_queries,
        seed=seed,
        recall_k=recall_k,
        device=device,
        node_features=x,
    )
