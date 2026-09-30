"""Connected BFS subsample of a transductive node-classification graph.

Starts at a random train node of a size-ranked class, expands in BFS order
until ``target_nodes``, then returns the induced subgraph. Labels that still
appear are remapped to ``0..K-1``; original ids are stored on the result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class ClassSplitRow:
    orig_class: int | None
    n: int
    train: int
    val: int
    test: int
    none: int


@dataclass
class BFSSubsample:
    data: Data
    node_index: Tensor
    hops: Tensor
    seed_node: int
    seed_class: int
    class_rank: int
    ranked_classes: Tensor
    full_class_sizes: Tensor
    orig_class_ids: Tensor
    n_full: int
    hit_budget: bool


def node_class_labels(y: Tensor) -> tuple[Tensor, Tensor]:
    """Return integer labels and a labeled-node mask.

    Unlabeled GraphLand nodes are ``nan``; they get label ``-1``.
    """
    if y.ndim == 2:
        labeled = y.abs().sum(dim=1) > 0
        labels = y.argmax(dim=1).long()
        labels = labels.masked_fill(~labeled, -1)
        return labels, labeled

    y1 = y.view(-1)
    if y1.dtype.is_floating_point:
        labeled = torch.isfinite(y1)
        labels = torch.full((y1.numel(),), -1, dtype=torch.long)
        labels[labeled] = y1[labeled].long()
        return labels, labeled
    return y1.long(), torch.ones(y1.numel(), dtype=torch.bool)


def class_sizes(labels: Tensor, labeled: Tensor, num_classes: int | None = None) -> Tensor:
    lab = labels[labeled]
    if lab.numel() == 0:
        return torch.zeros(0, dtype=torch.long)
    n_cls = int(lab.max().item()) + 1 if num_classes is None else int(num_classes)
    return torch.bincount(lab, minlength=n_cls)


def rank_classes(sizes: Tensor) -> Tensor:
    """Class ids sorted by size descending, then id ascending (stable ties)."""
    sizes_np = sizes.detach().cpu().numpy()
    cls = np.arange(sizes_np.shape[0])
    order = np.lexsort((cls, -sizes_np))
    return torch.from_numpy(order.astype(np.int64))


def pick_train_seed(
    train_mask: Tensor,
    labels: Tensor,
    labeled: Tensor,
    class_id: int,
    generator: torch.Generator,
) -> int:
    candidates = torch.where(train_mask & labeled & (labels == int(class_id)))[0]
    if candidates.numel() == 0:
        raise RuntimeError(
            f"No train nodes in class {class_id}; cannot draw a BFS seed."
        )
    idx = torch.randint(0, candidates.numel(), (1,), generator=generator)
    return int(candidates[idx].item())


def _csr_adjacency(edge_index: Tensor, num_nodes: int) -> csr_matrix:
    src = edge_index[0].detach().cpu().numpy()
    dst = edge_index[1].detach().cpu().numpy()
    ones = np.ones(src.shape[0], dtype=np.uint8)
    return csr_matrix((ones, (src, dst)), shape=(num_nodes, num_nodes))


def bfs_order(
    edge_index: Tensor,
    num_nodes: int,
    seed: int,
    target_nodes: int,
) -> tuple[Tensor, Tensor, bool]:
    """Visit up to ``target_nodes`` in BFS order from ``seed``.

    Returns ``(node_index, hops, hit_budget)``. ``hit_budget`` is True when the
    connected component was large enough to fill ``target_nodes``.
    """
    if target_nodes < 1:
        raise ValueError("target_nodes must be >= 1")
    if seed < 0 or seed >= num_nodes:
        raise ValueError(f"seed {seed} out of range for n={num_nodes}")

    adj = _csr_adjacency(edge_index, num_nodes)
    indptr = adj.indptr
    indices = adj.indices
    cap = min(int(target_nodes), int(num_nodes))

    seen = np.zeros(num_nodes, dtype=np.bool_)
    order = np.empty(cap, dtype=np.int64)
    hops = np.empty(cap, dtype=np.int32)
    seen[seed] = True
    order[0] = seed
    hops[0] = 0
    head = 0
    tail = 1

    while head < tail and tail < cap:
        u = int(order[head])
        h = int(hops[head])
        head += 1
        start = int(indptr[u])
        end = int(indptr[u + 1])
        for v in indices[start:end]:
            v = int(v)
            if seen[v]:
                continue
            seen[v] = True
            order[tail] = v
            hops[tail] = h + 1
            tail += 1
            if tail >= cap:
                break

    return (
        torch.from_numpy(order[:tail]),
        torch.from_numpy(hops[:tail]),
        bool(tail >= cap),
    )


def induced_subgraph(data: Data, keep: Tensor) -> Data:
    """Induced subgraph on ``keep`` (original ids), relabelled ``0..n-1``."""
    n = int(data.num_nodes)
    keep = keep.detach().cpu().long()
    ei, _ = subgraph(
        keep,
        data.edge_index,
        relabel_nodes=True,
        num_nodes=n,
    )
    out = Data(edge_index=ei, num_nodes=int(keep.numel()))
    keys_attr = getattr(data, "keys", None)
    if callable(keys_attr):
        key_iter = keys_attr()
    elif keys_attr is not None:
        key_iter = keys_attr
    else:
        key_iter = []
    for key in key_iter:
        if key in {"edge_index", "num_nodes"}:
            continue
        val = data[key]
        if not torch.is_tensor(val):
            continue
        if val.dim() >= 1 and val.size(0) == n:
            out[key] = val[keep]
        else:
            out[key] = val
    return out


def restrict_to_top_classes(data: Data, k: int) -> Data:
    """Keep labels and splits only for the ``k`` largest classes.

    Other nodes stay in the graph but become unlabeled and are removed from
    train / val / test. Ranking uses ``orig_y`` when present. Remaining labels
    are remapped to ``0..k-1``.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    y_src = getattr(data, "orig_y", None)
    if y_src is None:
        y_src = data.y
    labels, labeled = node_class_labels(y_src)
    sizes = class_sizes(labels, labeled)
    n_present = int((sizes > 0).sum().item())
    if n_present == 0:
        raise RuntimeError("Graph has no labeled nodes.")
    k_eff = min(int(k), n_present)
    ranked = rank_classes(sizes)
    keep_ids = ranked[:k_eff]
    keep = labeled & torch.isin(labels, keep_ids)

    orig_y = y_src.detach().clone()
    if not orig_y.dtype.is_floating_point:
        orig_y = orig_y.to(dtype=torch.float32)
        orig_y[~labeled] = float("nan")
    orig_y = orig_y.clone()
    orig_y[~keep] = float("nan")

    new_y, orig_class_ids = remap_present_classes(orig_y)
    out = data.clone()
    out.y = new_y
    out.orig_y = orig_y
    out.orig_class_ids = orig_class_ids
    out.train_mask = data.train_mask.clone() & keep
    out.val_mask = data.val_mask.clone() & keep
    out.test_mask = data.test_mask.clone() & keep
    out.keep_top_classes = int(k_eff)
    dropped = ranked[k_eff:]
    dropped = dropped[sizes[dropped] > 0]
    out.dropped_class_ids = dropped.long()
    return out


def remap_present_classes(y: Tensor) -> tuple[Tensor, Tensor]:
    """Map appearing class ids to ``0..K-1``. Unlabeled stay ``nan`` / ``-1``."""
    labels, labeled = node_class_labels(y)
    if not bool(labeled.any()):
        raise RuntimeError("Subsample has no labeled nodes.")
    present = torch.unique(labels[labeled])
    mapped = torch.full((int(present.max().item()) + 1,), -1, dtype=torch.long)
    mapped[present] = torch.arange(present.numel(), dtype=torch.long)
    if y.ndim == 2:
        raise ValueError("remap_present_classes expects 1-D labels")
    new_y = y.detach().clone()
    if new_y.dtype.is_floating_point:
        new_y[labeled] = mapped[labels[labeled]].to(dtype=new_y.dtype)
    else:
        new_y = mapped[labels]
        new_y = new_y.masked_fill(~labeled, -1)
    return new_y, present.long()


def class_split_rows(
    labels: Tensor,
    labeled: Tensor,
    train_mask: Tensor,
    val_mask: Tensor,
    test_mask: Tensor,
) -> list[ClassSplitRow]:
    """Per original class (plus unlabeled) counts by RL mask."""
    rows: list[ClassSplitRow] = []
    lab = labels[labeled]
    if lab.numel():
        for cid in torch.unique(lab).tolist():
            in_c = labeled & (labels == int(cid))
            n = int(in_c.sum().item())
            n_tr = int((in_c & train_mask).sum().item())
            n_va = int((in_c & val_mask).sum().item())
            n_te = int((in_c & test_mask).sum().item())
            rows.append(
                ClassSplitRow(
                    orig_class=int(cid),
                    n=n,
                    train=n_tr,
                    val=n_va,
                    test=n_te,
                    none=n - n_tr - n_va - n_te,
                )
            )
        rows.sort(key=lambda r: r.n, reverse=True)
    unlabeled = ~labeled
    n_u = int(unlabeled.sum().item())
    if n_u:
        n_tr = int((unlabeled & train_mask).sum().item())
        n_va = int((unlabeled & val_mask).sum().item())
        n_te = int((unlabeled & test_mask).sum().item())
        rows.append(
            ClassSplitRow(
                orig_class=None,
                n=n_u,
                train=n_tr,
                val=n_va,
                test=n_te,
                none=n_u - n_tr - n_va - n_te,
            )
        )
    return rows


def subsample_bfs(
    data: Data,
    *,
    target_nodes: int = 100_000,
    class_rank: int = 2,
    seed: int = 0,
) -> BFSSubsample:
    """BFS subsample starting at a random train node of the given class rank.

    ``class_rank=0`` is the largest labeled class on the full graph; ``2`` is
    the third-largest.
    """
    n = int(data.num_nodes)
    labels, labeled = node_class_labels(data.y)
    sizes = class_sizes(labels, labeled)
    ranked = rank_classes(sizes)
    if class_rank < 0 or class_rank >= ranked.numel():
        raise ValueError(
            f"class_rank={class_rank} out of range for {ranked.numel()} classes"
        )
    seed_class = int(ranked[class_rank].item())
    generator = torch.Generator().manual_seed(int(seed))
    seed_node = pick_train_seed(
        data.train_mask, labels, labeled, seed_class, generator
    )
    node_index, hops, hit_budget = bfs_order(
        data.edge_index, n, seed_node, target_nodes
    )
    sub = induced_subgraph(data, node_index)
    orig_y = sub.y.detach().clone()
    new_y, orig_class_ids = remap_present_classes(sub.y)
    sub.y = new_y
    sub.orig_y = orig_y
    sub.orig_node_index = node_index.long()
    sub.orig_class_ids = orig_class_ids
    sub.bfs_seed_node = int(seed_node)
    sub.bfs_seed_class = int(seed_class)
    sub.bfs_class_rank = int(class_rank)
    sub.bfs_seed = int(seed)
    sub.bfs_target_nodes = int(target_nodes)
    sub.bfs_max_hop = int(hops.max().item()) if hops.numel() else 0
    sub.bfs_hit_budget = bool(hit_budget)
    sub.full_num_nodes = int(n)
    sub.full_class_sizes = sizes.long()
    sub.full_ranked_classes = ranked.long()
    return BFSSubsample(
        data=sub,
        node_index=node_index.long(),
        hops=hops,
        seed_node=seed_node,
        seed_class=seed_class,
        class_rank=int(class_rank),
        ranked_classes=ranked.long(),
        full_class_sizes=sizes.long(),
        orig_class_ids=orig_class_ids,
        n_full=n,
        hit_budget=bool(hit_budget),
    )
