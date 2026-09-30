"""Per-source sampled Recall@K for a shared Wander / UniLP comparison.

Each undirected eval edge becomes two directed queries (u→v and v→u). For
source ``u``, candidates are that source's eval tails plus ``M`` sampled
non-edges (train/val/test neighbors and self excluded). Both models rank the
same cached lists; metrics are macro-averaged over sources.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

CACHE_VERSION = 1
DEFAULT_NUM_NEG = 100
SPLIT_ALIASES = {"val": "valid", "valid": "valid", "test": "test"}
# Independent of the negative-sampling seed (10007 * run + …).
SOURCE_SUBSAMPLE_SEED_BASE = 17_777


def sampled_recall_cache_path(
    dataset_dir: str | Path,
    run: int,
    split: str,
    num_neg: int = DEFAULT_NUM_NEG,
) -> Path:
    split = SPLIT_ALIASES.get(split, split)
    return Path(dataset_dir) / f"recall_cands{int(run)}_{split}_m{int(num_neg)}.pt"


def _pairs_e2(pairs: Optional[Tensor]) -> Tensor:
    if pairs is None:
        return torch.empty((0, 2), dtype=torch.long)
    t = pairs.detach().cpu().long()
    if t.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long)
    return t.reshape(-1, 2)


def _undirected_adj(pairs: Tensor, num_nodes: int) -> List[set[int]]:
    adj: List[set[int]] = [set() for _ in range(num_nodes)]
    for u, v in _pairs_e2(pairs).tolist():
        u, v = int(u), int(v)
        if u == v or u < 0 or v < 0 or u >= num_nodes or v >= num_nodes:
            continue
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _csr_from_lists(rows: Sequence[Sequence[int]]) -> Tuple[Tensor, Tensor]:
    ptr = [0]
    tails: List[int] = []
    for row in rows:
        tails.extend(int(x) for x in row)
        ptr.append(len(tails))
    return (
        torch.tensor(ptr, dtype=torch.long),
        torch.tensor(tails, dtype=torch.long) if tails else torch.empty(0, dtype=torch.long),
    )


def build_sampled_recall_cache(
    split_edge: Dict[str, Dict[str, Tensor]],
    num_nodes: int,
    *,
    split: str = "test",
    run: int = 0,
    num_neg: int = DEFAULT_NUM_NEG,
    bidirectional: bool = True,
    sample_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a per-source candidate cache from a UniLP 70/10/20 ``split_edge``."""
    split = SPLIT_ALIASES.get(split, split)
    if split not in ("valid", "test"):
        raise ValueError(f"sampled recall split must be valid/test, got {split!r}")
    num_neg = int(num_neg)
    if num_neg < 1:
        raise ValueError(f"num_neg must be >= 1, got {num_neg}")
    if sample_seed is None:
        sample_seed = 10_007 * int(run) + (1 if split == "test" else 0)

    train_adj = _undirected_adj(split_edge["train"]["edge"], num_nodes)
    val_adj = _undirected_adj(split_edge["valid"]["edge"], num_nodes)
    test_adj = _undirected_adj(split_edge["test"]["edge"], num_nodes)
    eval_pairs = _pairs_e2(split_edge[split]["edge"])
    if bidirectional:
        eval_out = test_adj if split == "test" else val_adj
    else:
        eval_out = [set() for _ in range(num_nodes)]
        for u, v in eval_pairs.tolist():
            eval_out[int(u)].add(int(v))

    sources = sorted(u for u in range(num_nodes) if eval_out[u])
    if not sources:
        raise ValueError(f"no {split} edges to group by source")

    g = torch.Generator().manual_seed(int(sample_seed))
    pos_rows: List[List[int]] = []
    neg_rows: List[List[int]] = []
    for u in sources:
        pos = sorted(eval_out[u])
        forbidden = {u} | train_adj[u] | val_adj[u] | test_adj[u]
        pool = [i for i in range(num_nodes) if i not in forbidden]
        if not pool:
            raise ValueError(f"source {u} has no non-edges to sample")
        take = min(num_neg, len(pool))
        perm = torch.randperm(len(pool), generator=g)[:take]
        negs = sorted(int(pool[int(i)]) for i in perm.tolist())
        pos_rows.append(pos)
        neg_rows.append(negs)

    pos_ptr, pos_tails = _csr_from_lists(pos_rows)
    neg_ptr, neg_tails = _csr_from_lists(neg_rows)
    return {
        "version": CACHE_VERSION,
        "num_nodes": int(num_nodes),
        "num_neg": num_neg,
        "bidirectional": bool(bidirectional),
        "run": int(run),
        "split": split,
        "sample_seed": int(sample_seed),
        "sources": torch.tensor(sources, dtype=torch.long),
        "pos_ptr": pos_ptr,
        "pos_tails": pos_tails,
        "neg_ptr": neg_ptr,
        "neg_tails": neg_tails,
    }


def save_sampled_recall_cache(path: str | Path, cache: Dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    return path


def load_sampled_recall_cache(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    try:
        cache = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        cache = torch.load(path, map_location="cpu")
    if int(cache.get("version", -1)) != CACHE_VERSION:
        raise ValueError(f"unsupported sampled-recall cache version in {path}")
    return cache


def load_or_build_sampled_recall_cache(
    dataset_dir: str | Path,
    split_edge: Dict[str, Dict[str, Tensor]],
    num_nodes: int,
    *,
    run: int,
    split: str,
    num_neg: int = DEFAULT_NUM_NEG,
    bidirectional: bool = True,
) -> Dict[str, Any]:
    path = sampled_recall_cache_path(dataset_dir, run, split, num_neg)
    if path.is_file():
        cache = load_sampled_recall_cache(path)
        if int(cache["num_neg"]) != int(num_neg) or cache["split"] != SPLIT_ALIASES.get(split, split):
            raise ValueError(f"cache mismatch at {path}")
        return cache
    cache = build_sampled_recall_cache(
        split_edge,
        num_nodes,
        split=split,
        run=run,
        num_neg=num_neg,
        bidirectional=bidirectional,
    )
    save_sampled_recall_cache(path, cache)
    print(f"save sampled-recall cache to {path}")
    return cache


def source_slice(cache: Dict[str, Any], i: int, kind: str) -> Tensor:
    ptr = cache[f"{kind}_ptr"]
    tails = cache[f"{kind}_tails"]
    return tails[int(ptr[i]) : int(ptr[i + 1])]


def source_subsample_seed(run: int, split: str) -> int:
    split = SPLIT_ALIASES.get(split, split)
    return SOURCE_SUBSAMPLE_SEED_BASE * int(run) + (1 if split == "test" else 0)


def subsample_sampled_recall_cache(
    cache: Dict[str, Any],
    max_sources: int,
    *,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Keep a deterministic subset of sources. Same seed → same IDs for both models."""
    n = int(cache["sources"].numel())
    take = int(max_sources)
    if take < 1:
        raise ValueError(f"max_sources must be >= 1, got {max_sources}")
    if take >= n:
        return cache
    if seed is None:
        seed = source_subsample_seed(int(cache.get("run", 0)), str(cache.get("split", "test")))
    g = torch.Generator().manual_seed(int(seed))
    keep = torch.sort(torch.randperm(n, generator=g)[:take]).values
    pos_rows = [source_slice(cache, int(i), "pos").tolist() for i in keep.tolist()]
    neg_rows = [source_slice(cache, int(i), "neg").tolist() for i in keep.tolist()]
    pos_ptr, pos_tails = _csr_from_lists(pos_rows)
    neg_ptr, neg_tails = _csr_from_lists(neg_rows)
    out = dict(cache)
    out["sources"] = cache["sources"].long()[keep]
    out["pos_ptr"] = pos_ptr
    out["pos_tails"] = pos_tails
    out["neg_ptr"] = neg_ptr
    out["neg_tails"] = neg_tails
    out["max_sources"] = take
    out["source_subsample_seed"] = int(seed)
    return out


def flatten_sampled_recall_pairs(cache: Dict[str, Any]) -> Tuple[Tensor, Tensor]:
    """Directed ``[P, 2]`` positives and ``[N, 2]`` negatives, source-major order."""
    sources = cache["sources"].long()
    pos_rows: List[List[int]] = []
    neg_rows: List[List[int]] = []
    for i, src in enumerate(sources.tolist()):
        for t in source_slice(cache, i, "pos").tolist():
            pos_rows.append([int(src), int(t)])
        for t in source_slice(cache, i, "neg").tolist():
            neg_rows.append([int(src), int(t)])
    pos = torch.tensor(pos_rows, dtype=torch.long) if pos_rows else torch.empty((0, 2), dtype=torch.long)
    neg = torch.tensor(neg_rows, dtype=torch.long) if neg_rows else torch.empty((0, 2), dtype=torch.long)
    return pos, neg


def source_recall_ndcg(
    pos_scores: Tensor,
    neg_scores: Tensor,
    ks: Sequence[int] = (20, 50),
) -> Dict[str, float]:
    """Recall@K / NDCG@K for one source on its pos ∪ sampled-neg list."""
    pos = pos_scores.detach().float().reshape(-1)
    neg = neg_scores.detach().float().reshape(-1)
    n_pos = int(pos.numel())
    if n_pos == 0:
        return {f"recall@{k}": 0.0 for k in ks} | {f"ndcg@{k}": 0.0 for k in ks}
    scores = torch.cat([pos, neg], dim=0)
    n = int(scores.numel())
    order = torch.argsort(scores, descending=True)
    rank = torch.empty(n, dtype=torch.long)
    rank[order] = torch.arange(n)
    pos_ranks = rank[:n_pos]
    out: Dict[str, float] = {}
    for k in ks:
        kk = min(int(k), n)
        hits = int((pos_ranks < kk).sum().item())
        out[f"recall@{k}"] = hits / n_pos
        dcg = 0.0
        for r in pos_ranks.tolist():
            if r < kk:
                dcg += 1.0 / math.log2(r + 2)
        max_dcg = sum(1.0 / math.log2(r + 2) for r in range(min(n_pos, kk)))
        out[f"ndcg@{k}"] = (dcg / max_dcg) if max_dcg > 0 else 0.0
    return out


def evaluate_sampled_recall(
    cache: Dict[str, Any],
    pos_scores: Tensor,
    neg_scores: Tensor,
    ks: Sequence[int] = (20, 50),
) -> Dict[str, float]:
    """Macro-average Recall@K / NDCG@K from flattened pos-then-neg scores.

    ``pos_scores`` / ``neg_scores`` must follow :func:`flatten_sampled_recall_pairs`.
    """
    pos_scores = pos_scores.detach().float().reshape(-1)
    neg_scores = neg_scores.detach().float().reshape(-1)
    n_src = int(cache["sources"].numel())
    sums = {f"recall@{k}": 0.0 for k in ks}
    sums.update({f"ndcg@{k}": 0.0 for k in ks})
    pos_off = 0
    neg_off = 0
    n_used = 0
    for i in range(n_src):
        n_pos = int(cache["pos_ptr"][i + 1] - cache["pos_ptr"][i])
        n_neg = int(cache["neg_ptr"][i + 1] - cache["neg_ptr"][i])
        if n_pos == 0:
            pos_off += n_pos
            neg_off += n_neg
            continue
        row = source_recall_ndcg(
            pos_scores[pos_off : pos_off + n_pos],
            neg_scores[neg_off : neg_off + n_neg],
            ks=ks,
        )
        for key, val in row.items():
            sums[key] += val
        n_used += 1
        pos_off += n_pos
        neg_off += n_neg
    if pos_off != int(pos_scores.numel()) or neg_off != int(neg_scores.numel()):
        raise ValueError(
            f"score/cache length mismatch: used pos={pos_off}/{pos_scores.numel()} "
            f"neg={neg_off}/{neg_scores.numel()}"
        )
    denom = n_used if n_used else 1
    out = {key: val / denom for key, val in sums.items()}
    out["num_sources"] = float(n_used)
    return out


def iter_source_candidates(
    cache: Dict[str, Any],
    source_indices: Iterable[int],
) -> List[Tuple[int, Tensor, Tensor]]:
    """``(global_source_id, pos_tails, neg_tails)`` for loader rows."""
    sources = cache["sources"].long()
    out: List[Tuple[int, Tensor, Tensor]] = []
    for i in source_indices:
        i = int(i)
        out.append(
            (
                int(sources[i].item()),
                source_slice(cache, i, "pos"),
                source_slice(cache, i, "neg"),
            )
        )
    return out
