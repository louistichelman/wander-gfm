"""Planted disjoint C_k motifs on a d-regular backbone.

Used by the fixed CYCLES eval graph. Labels are planted-motif membership;
degrees are exactly ``d`` for every node (default d=3) so degree is not a
shortcut.

Backbone choice by motif length ``k``:
- k=3,5: bipartite configuration model (no odd cycles in the backbone alone)
- k=4: non-bipartite stub matching with girth >= 5 while adding edges
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from . import motif_er


DEFAULT_DEGREE = 3
DEFAULT_TARGET_POS_RATE = 0.25
MAX_BUILD_TRIES = 48
MAX_GIRTH_MATCH_ROUNDS = 256

MOTIF_TYPE_TO_K: Dict[str, int] = {
    "triangle": 3,
    "c4": 4,
    "c5": 5,
}

_MOTIF_DETECT: Dict[int, Callable[[torch.Tensor, int], torch.Tensor]] = {
    3: motif_er.nodes_in_triangles,
    4: motif_er.nodes_in_c4,
    5: motif_er.nodes_in_c5,
}


def n_motifs_for_pos_rate(n: int, k: int, pos_rate: float) -> int:
    """Choose motif count ``m`` near ``pos_rate * n / k`` with a valid backbone.

    For bipartite backbones (k in {3,5}), ``k*m`` must be even (so ``m`` even when
    ``k`` is odd) and the backbone size must be even. For k=4 the backbone is
    non-bipartite; we still require even ``n`` and ``k*m <= n_back``.
    """
    if k not in (3, 4, 5):
        raise ValueError(f"k must be 3, 4, or 5, got {k}")
    if n < 2 * k or n % 2 != 0:
        raise ValueError(f"n must be even and >= {2 * k}, got {n}")

    bipartite = k in (3, 5)

    def _ok(m: int) -> bool:
        if m < 1:
            return False
        if bipartite and (m < 2 or m % 2 != 0):
            return False
        if k == 4 and m < 1:
            return False
        n_planted = k * m
        n_back = n - n_planted
        if n_back < 2 * DEFAULT_DEGREE or n_planted > n_back:
            return False
        if bipartite and n_back % 2 != 0:
            return False
        return True

    m = int(round(pos_rate * n / float(k)))
    if bipartite and m % 2 == 1:
        lo, hi = m - 1, m + 1
        rate_lo = (k * lo) / n if lo >= 2 else -1.0
        rate_hi = (k * hi) / n
        m = hi if abs(rate_hi - pos_rate) < abs(rate_lo - pos_rate) else lo
    m = max(m, 2 if bipartite else 1)
    if _ok(m):
        return m
    step = 2 if bipartite else 1
    for delta in range(step, n // k + step, step):
        for mm in (m - delta, m + delta):
            if _ok(mm):
                return mm
    raise ValueError(
        f"cannot fit planted C{k} into n={n} at pos_rate={pos_rate}"
    )


def n_triangles_for_pos_rate(n: int, pos_rate: float) -> int:
    """Back-compat wrapper for planted triangles."""
    return n_motifs_for_pos_rate(n, 3, pos_rate)


def _rand_perm(n: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randperm(n, generator=generator)


def _configuration_bipartite(
    left_stubs: list[int],
    right_stubs: list[int],
    generator: torch.Generator,
    *,
    max_tries: int = 64,
) -> Optional[list[Tuple[int, int]]]:
    """Match left stubs to right stubs; return undirected edges or None."""
    if len(left_stubs) != len(right_stubs):
        return None
    if not left_stubs:
        return []
    n_stub = len(left_stubs)
    for _ in range(max_tries):
        order = _rand_perm(n_stub, generator)
        right = [right_stubs[int(i)] for i in order.tolist()]
        edges: set[Tuple[int, int]] = set()
        ok = True
        for a, b in zip(left_stubs, right):
            if a == b:
                ok = False
                break
            e = (a, b) if a < b else (b, a)
            if e in edges:
                ok = False
                break
            edges.add(e)
        if ok:
            return list(edges)
    return None


def _bfs_dist(adj: List[Set[int]], src: int, dst: int, max_depth: int) -> int:
    """Shortest-path distance, or ``max_depth + 1`` if farther / unreachable."""
    if src == dst:
        return 0
    if dst in adj[src]:
        return 1
    q: deque[Tuple[int, int]] = deque([(src, 0)])
    seen = {src}
    while q:
        u, dist = q.popleft()
        if dist >= max_depth:
            continue
        for v in adj[u]:
            if v in seen:
                continue
            if v == dst:
                return dist + 1
            seen.add(v)
            q.append((v, dist + 1))
    return max_depth + 1


def _configuration_girth(
    stubs: list[int],
    adj: List[Set[int]],
    generator: torch.Generator,
    *,
    forbid_dist_le: int = 3,
    max_rounds: int = MAX_GIRTH_MATCH_ROUNDS,
) -> Optional[list[Tuple[int, int]]]:
    """Pair stubs while forbidding edges that close a cycle of length <= forbid_dist_le+1.

    For girth >= 5 use ``forbid_dist_le=3`` (no edge if current dist <= 3).
    Mutates ``adj`` on success; on failure the caller must discard the attempt.
    """
    unmatched = list(stubs)
    new_edges: list[Tuple[int, int]] = []
    for _ in range(max_rounds):
        if not unmatched:
            return new_edges
        order = _rand_perm(len(unmatched), generator)
        unmatched = [unmatched[int(i)] for i in order.tolist()]
        still: list[int] = []
        i = 0
        progress = False
        while i + 1 < len(unmatched):
            u = unmatched[i]
            v = unmatched[i + 1]
            i += 2
            if u == v or v in adj[u]:
                still.extend([u, v])
                continue
            if _bfs_dist(adj, u, v, forbid_dist_le) <= forbid_dist_le:
                still.extend([u, v])
                continue
            adj[u].add(v)
            adj[v].add(u)
            e = (u, v) if u < v else (v, u)
            new_edges.append(e)
            progress = True
        if i < len(unmatched):
            still.append(unmatched[i])
        if not progress and still:
            return None
        unmatched = still
    return None if unmatched else new_edges


def _edges_to_edge_index(edges: list[Tuple[int, int]], n: int) -> torch.Tensor:
    if not edges:
        return torch.zeros(2, 0, dtype=torch.long)
    src = []
    dst = []
    for u, v in edges:
        src.extend([u, v])
        dst.extend([v, u])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return to_undirected(edge_index, num_nodes=n)


def _degrees(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    deg = torch.zeros(n, dtype=torch.long)
    if edge_index.numel() == 0:
        return deg
    ones = torch.ones(edge_index.size(1), dtype=torch.long)
    deg.scatter_add_(0, edge_index[0], ones)
    return deg


def _plant_cycle_edges(
    k: int,
    n_motifs: int,
    *,
    node_offset: int = 0,
) -> list[Tuple[int, int]]:
    edges: list[Tuple[int, int]] = []
    for t in range(n_motifs):
        base = node_offset + k * t
        nodes = [base + i for i in range(k)]
        for i in range(k):
            a, b = nodes[i], nodes[(i + 1) % k]
            edges.append((a, b) if a < b else (b, a))
    return edges


def n_motifs_mixed_balanced(
    n: int,
    total_pos_rate: float = DEFAULT_TARGET_POS_RATE,
    *,
    nodes_per_class: int | None = None,
) -> Tuple[int, int, int]:
    """Return ``(m3, m4, m5)`` with ~equal planted nodes per class.

    When ``nodes_per_class`` is set (e.g. 600), each motif class occupies
    exactly ``nodes_per_class`` nodes: ``m3 = N/3``, ``m4 = N/4``, ``m5 = N/5``.

    Otherwise derive from ``total_pos_rate``; for ``n=5000`` and rate 0.25 this
    yields ``(139, 104, 83)`` → 417 / 416 / 415 planted nodes.
    """
    if n < 30 or n % 2 != 0:
        raise ValueError(f"n must be even and >= 30, got {n}")
    if nodes_per_class is not None:
        if nodes_per_class < 5:
            raise ValueError(f"nodes_per_class must be >= 5, got {nodes_per_class}")
        m3 = nodes_per_class // 3
        m4 = nodes_per_class // 4
        m5 = nodes_per_class // 5
    else:
        per_class = int(round(total_pos_rate * n / 3.0))
        m3 = max(1, per_class // 3)
        m4 = max(1, per_class // 4)
        m5 = max(1, per_class // 5)
    n_planted = 3 * m3 + 4 * m4 + 5 * m5
    n_back = n - n_planted
    if n_back < 2 * DEFAULT_DEGREE or n_planted > n_back:
        raise ValueError(
            f"cannot fit balanced mixed motifs into n={n}: "
            f"m3={m3}, m4={m4}, m5={m5}, n_planted={n_planted}, n_back={n_back}"
        )
    return m3, m4, m5


def sample_planted_ck_regular(
    n: int,
    d: int,
    k: int,
    n_motifs: int,
    generator: torch.Generator,
    *,
    max_tries: int = MAX_BUILD_TRIES,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample planted-C_k / d-regular graph.

    Returns
    -------
    edge_index
        Undirected bidirectional ``[2, E]``.
    y
        Long labels: 1 iff node belongs to a planted C_k.
    """
    if d != DEFAULT_DEGREE:
        raise ValueError(f"only d={DEFAULT_DEGREE} supported in v1, got {d}")
    if k not in (3, 4, 5):
        raise ValueError(f"k must be 3, 4, or 5, got {k}")
    if n % 2 != 0 or n < 2 * k:
        raise ValueError(f"n must be even and >= {2 * k}, got {n}")
    bipartite = k in (3, 5)
    if bipartite:
        if n_motifs < 2 or n_motifs % 2 != 0:
            raise ValueError(f"n_motifs must be even and >= 2 for k={k}, got {n_motifs}")
    elif n_motifs < 1:
        raise ValueError(f"n_motifs must be >= 1, got {n_motifs}")

    n_planted = k * n_motifs
    n_back = n - n_planted
    if n_back < 2 * d or n_planted > n_back:
        raise ValueError(
            f"invalid sizes n_back={n_back}, n_planted={n_planted} for n={n}, m={n_motifs}"
        )
    if bipartite and n_back % 2 != 0:
        raise ValueError(f"bipartite backbone needs even n_back, got {n_back}")

    detect = _MOTIF_DETECT[k]
    planted_y = torch.zeros(n, dtype=torch.long)
    planted_y[:n_planted] = 1
    planted_set = set(range(n_planted))
    back_ids = list(range(n_planted, n))

    for _attempt in range(max_tries):
        edges: list[Tuple[int, int]] = _plant_cycle_edges(k, n_motifs)

        if bipartite:
            n_left = n_back // 2
            left_ids = back_ids[:n_left]
            right_ids = back_ids[n_left:]
            n_bridge_l = n_planted // 2
            n_bridge_r = n_planted - n_bridge_l
            left_order = _rand_perm(len(left_ids), generator)
            right_order = _rand_perm(len(right_ids), generator)
            attach_l = [left_ids[int(i)] for i in left_order[:n_bridge_l].tolist()]
            attach_r = [right_ids[int(i)] for i in right_order[:n_bridge_r].tolist()]
            attach = attach_l + attach_r
        else:
            left_ids = []
            right_ids = []
            attach_order = _rand_perm(n_back, generator)
            attach = [back_ids[int(i)] for i in attach_order[:n_planted].tolist()]

        plant_order = _rand_perm(n_planted, generator).tolist()
        attach_perm = _rand_perm(n_planted, generator).tolist()
        bridges = [
            (plant_order[i], attach[attach_perm[i]])
            for i in range(n_planted)
        ]
        for u, v in bridges:
            edges.append((u, v) if u < v else (v, u))

        bridge_count = {u: 0 for u in back_ids}
        for _, v in bridges:
            bridge_count[v] += 1
        if any(bridge_count[v] > d for v in bridge_count):
            continue

        if bipartite:
            left_stubs: list[int] = []
            right_stubs: list[int] = []
            for u in left_ids:
                left_stubs.extend([u] * (d - bridge_count[u]))
            for v in right_ids:
                right_stubs.extend([v] * (d - bridge_count[v]))
            if len(left_stubs) != len(right_stubs):
                continue
            back_edges = _configuration_bipartite(left_stubs, right_stubs, generator)
            if back_edges is None:
                continue
            edges.extend(back_edges)
        else:
            # Build adjacency from planted cycles + bridges, then girth-fill.
            adj: List[Set[int]] = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            stubs: list[int] = []
            for u in back_ids:
                stubs.extend([u] * (d - bridge_count[u]))
            # Planted nodes already have degree 3 (2 cycle + 1 bridge).
            back_edges = _configuration_girth(stubs, adj, generator)
            if back_edges is None:
                continue
            edges.extend(back_edges)

        edge_set = {(u, v) if u < v else (v, u) for u, v in edges}
        edge_index = _edges_to_edge_index(sorted(edge_set), n)
        deg = _degrees(edge_index, n)
        if not bool(torch.all(deg == d)):
            continue

        in_motif = detect(edge_index, n)
        if set(int(i) for i in in_motif.nonzero(as_tuple=True)[0].tolist()) != planted_set:
            continue

        return edge_index, planted_y.clone()

    raise RuntimeError(
        f"failed to sample planted C{k} regular graph after {max_tries} tries "
        f"(n={n}, d={d}, m={n_motifs})"
    )


def sample_planted_mixed_regular(
    n: int,
    d: int,
    m3: int,
    m4: int,
    m5: int,
    generator: torch.Generator,
    *,
    max_tries: int = MAX_BUILD_TRIES,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a 3-regular graph with disjoint planted C3 / C4 / C5 blocks.

    Labels: ``0`` backbone, ``1`` triangle, ``2`` C4, ``3`` C5.
    Uses a girth>=5 backbone (same strategy as single-type C4 graphs).
    """
    if d != DEFAULT_DEGREE:
        raise ValueError(f"only d={DEFAULT_DEGREE} supported, got {d}")
    if n % 2 != 0:
        raise ValueError(f"n must be even, got {n}")
    for name, m in ("m3", m3), ("m4", m4), ("m5", m5):
        if m < 1:
            raise ValueError(f"{name} must be >= 1, got {m}")

    n3 = 3 * m3
    n4 = 4 * m4
    n5 = 5 * m5
    off4 = n3
    off5 = n3 + n4
    n_planted = n3 + n4 + n5
    n_back = n - n_planted
    if n_back < 2 * d or n_planted > n_back:
        raise ValueError(
            f"invalid mixed sizes n={n}, n_planted={n_planted}, n_back={n_back}"
        )

    c3_set = set(range(0, n3))
    c4_set = set(range(off4, off4 + n4))
    c5_set = set(range(off5, off5 + n5))
    planted_set = c3_set | c4_set | c5_set
    back_ids = list(range(n_planted, n))

    y = torch.zeros(n, dtype=torch.long)
    y[list(c3_set)] = 1
    y[list(c4_set)] = 2
    y[list(c5_set)] = 3

    block_checks = (
        (3, c3_set, m3, 0),
        (4, c4_set, m4, off4),
        (5, c5_set, m5, off5),
    )

    for _attempt in range(max_tries):
        edges: list[Tuple[int, int]] = []
        for k, _block, m, offset in block_checks:
            edges.extend(_plant_cycle_edges(k, m, node_offset=offset))

        attach_order = _rand_perm(n_back, generator)
        attach = [back_ids[int(i)] for i in attach_order[:n_planted].tolist()]
        plant_order = _rand_perm(n_planted, generator).tolist()
        attach_perm = _rand_perm(n_planted, generator).tolist()
        bridges = [
            (plant_order[i], attach[attach_perm[i]])
            for i in range(n_planted)
        ]
        for u, v in bridges:
            edges.append((u, v) if u < v else (v, u))

        bridge_count = {u: 0 for u in back_ids}
        for _, v in bridges:
            bridge_count[v] += 1
        if any(bridge_count[v] > d for v in bridge_count):
            continue

        adj: List[Set[int]] = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        stubs: list[int] = []
        for u in back_ids:
            stubs.extend([u] * (d - bridge_count[u]))
        back_edges = _configuration_girth(stubs, adj, generator)
        if back_edges is None:
            continue
        edges.extend(back_edges)

        edge_set = {(u, v) if u < v else (v, u) for u, v in edges}
        edge_index = _edges_to_edge_index(sorted(edge_set), n)
        deg = _degrees(edge_index, n)
        if not bool(torch.all(deg == d)):
            continue

        ok = True
        for k, block, _m, _off in block_checks:
            detected = _MOTIF_DETECT[k](edge_index, n)
            detected_set = set(int(i) for i in detected.nonzero(as_tuple=True)[0].tolist())
            if detected_set != block:
                ok = False
                break
        if not ok:
            continue

        return edge_index, y.clone()

    raise RuntimeError(
        f"failed to sample planted mixed graph after {max_tries} tries "
        f"(n={n}, d={d}, m3={m3}, m4={m4}, m5={m5})"
    )


def sample_planted_k3_regular(
    n: int,
    d: int,
    n_triangles: int,
    generator: torch.Generator,
    *,
    max_tries: int = MAX_BUILD_TRIES,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Back-compat wrapper for planted triangles."""
    return sample_planted_ck_regular(
        n, d, 3, n_triangles, generator, max_tries=max_tries
    )


def build_planted_motif_data(
    *,
    n: int,
    k: int,
    n_motifs: int,
    seed: int,
    d: int = DEFAULT_DEGREE,
    feat_dim: int = motif_er.FEAT_DIM,
    n_train: int = motif_er.TRAIN_SIZE,
    n_val: int | None = None,
) -> Data:
    """Fixed-eval PyG graph: planted C_k, random features, node split."""
    g = torch.Generator().manual_seed(seed)
    edge_index, y = sample_planted_ck_regular(n, d, k, n_motifs, g)
    x = torch.randn(n, feat_dim, generator=g)
    train_mask, val_mask, test_mask = motif_er.make_random_masks(
        n, g, n_train=n_train, n_val=n_val,
    )
    return Data(
        x=x,
        y=y,
        edge_index=edge_index,
        num_nodes=n,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        motif_degree=d,
        motif_k=k,
        n_motifs=n_motifs,
        n_triangles=n_motifs if k == 3 else 0,
    )


def build_planted_mixed_data(
    *,
    n: int,
    m3: int,
    m4: int,
    m5: int,
    seed: int,
    d: int = DEFAULT_DEGREE,
    feat_dim: int = 0,
    n_train: int = motif_er.TRAIN_SIZE,
    n_val: int | None = None,
) -> Data:
    """Fixed-eval PyG graph: planted C3/C4/C5 blocks, multi-class labels.

    Node features default to none (``feat_dim=0``). Positive ``feat_dim``
    yields constant-one columns (non-informative). The historical N(0, 1)
    draw is still consumed so train/val/test masks match the previous
    random-feature cache.
    """
    g = torch.Generator().manual_seed(seed)
    edge_index, y = sample_planted_mixed_regular(n, d, m3, m4, m5, g)
    _ = torch.randn(n, motif_er.FEAT_DIM, generator=g)
    x = (
        torch.ones(n, feat_dim, dtype=torch.float32)
        if feat_dim > 0
        else None
    )
    train_mask, val_mask, test_mask = motif_er.make_random_masks(
        n, g, n_train=n_train, n_val=n_val,
    )
    return Data(
        x=x,
        y=y,
        edge_index=edge_index,
        num_nodes=n,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        motif_degree=d,
        num_classes=4,
        n_motifs_c3=m3,
        n_motifs_c4=m4,
        n_motifs_c5=m5,
    )


def build_planted_triangle_data(
    *,
    n: int,
    n_triangles: int,
    seed: int,
    d: int = DEFAULT_DEGREE,
    feat_dim: int = motif_er.FEAT_DIM,
) -> Data:
    """Back-compat wrapper for planted triangles."""
    return build_planted_motif_data(
        n=n, k=3, n_motifs=n_triangles, seed=seed, d=d, feat_dim=feat_dim
    )


def sample_planted_k3_edge_index(
    n: int,
    d: int,
    n_triangles: int,
    *,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper with optional integer seed (prior path)."""
    return sample_planted_ck_edge_index(
        n, d, 3, n_triangles, seed=seed
    )


def sample_planted_ck_edge_index(
    n: int,
    d: int,
    k: int,
    n_motifs: int,
    *,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper with optional integer seed (prior path)."""
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    else:
        g.manual_seed(int(torch.randint(0, 2**31 - 1, (1,)).item()))
    return sample_planted_ck_regular(n, d, k, n_motifs, g)
