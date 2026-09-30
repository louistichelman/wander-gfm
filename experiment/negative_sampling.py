"""Negative sampling for Wander."""

from typing import Set, Tuple
from torch import Tensor
from torch_geometric.data import Data
import torch


def sample_negative_entities(
    batch: Tensor,
    data: Data,
    num_negatives: int,
    device: torch.device,
) -> Tensor:
    """Sample negative entity candidates for tail or head link-prediction queries.

    When ``batch`` has a fourth column, ``predict_head=1`` rows sample negative
    heads (filter ``(neg, r, t)``); ``predict_head=0`` rows sample negative tails
    (filter ``(h, r, neg)``). With a 3-column batch, all rows are tail queries.

    Args:
        batch: [batch_size, 3] or [batch_size, 4] with (head, rel, tail[, predict_head]).
        data: Data object with ``num_nodes`` and ``edge_set``.
        num_negatives: Number of negative samples per query.
        device: Device to create tensors on.

    Returns:
        negative_indices: [batch_size, num_negatives] sampled negative entity indices.
    """
    query_heads = batch[:, 0]
    query_relations = batch[:, 1]
    query_tails = batch[:, 2]
    if batch.shape[1] >= 4:
        predict_head = batch[:, 3].bool()
    else:
        predict_head = torch.zeros(batch.shape[0], dtype=torch.bool, device=batch.device)

    num_nodes = int(data.num_nodes)
    batch_size = batch.shape[0]

    if num_nodes == 0:
        return torch.zeros(batch_size, num_negatives, dtype=torch.long, device=device)

    # Bipartite candidate restriction: tail (item) predictions draw from
    # ``[offset, num_nodes)`` and head (user) predictions from ``[0, offset)``.
    # Non-bipartite graphs draw from the full node range.
    offset = 0
    if bool(getattr(data, "anygraph_bipartite", False)):
        offset = int(getattr(data, "anygraph_candidate_offset", 0))
    if 0 < offset < num_nodes:
        # Per-row [lo, hi) candidate range depending on predict_head.
        lo = torch.where(predict_head, 0, offset)
        hi = torch.where(predict_head, offset, num_nodes)
    else:
        lo = torch.zeros(batch_size, dtype=torch.long, device=batch.device)
        hi = torch.full((batch_size,), num_nodes, dtype=torch.long, device=batch.device)

    lo = lo.to(device)
    hi = hi.to(device)
    span = (hi - lo).clamp(min=1)
    rand = torch.rand(batch_size, num_negatives, device=device)
    result = (lo[:, None] + (rand * span[:, None]).long()).clamp(max=num_nodes - 1)

    edge_set = data.edge_set
    heads_list = query_heads.cpu().tolist()
    rels_list = query_relations.cpu().tolist()
    tails_list = query_tails.cpu().tolist()
    predict_head_list = predict_head.cpu().tolist()
    lo_list = lo.cpu().tolist()
    span_list = span.cpu().tolist()
    result_cpu = result.cpu()

    if edge_set is not None:
        for i in range(batch_size):
            r = rels_list[i]
            row_lo = lo_list[i]
            row_span = span_list[i]
            for j in range(num_negatives):
                neg = int(result_cpu[i, j].item())
                if predict_head_list[i]:
                    is_false_negative = (neg, r, tails_list[i]) in edge_set
                else:
                    is_false_negative = (heads_list[i], r, neg) in edge_set
                if is_false_negative:
                    for _ in range(100):
                        new_neg = row_lo + int(torch.randint(0, row_span, (1,)).item())
                        if predict_head_list[i]:
                            ok = (new_neg, r, tails_list[i]) not in edge_set
                        else:
                            ok = (heads_list[i], r, new_neg) not in edge_set
                        if ok:
                            result[i, j] = new_neg
                            break

    return result
