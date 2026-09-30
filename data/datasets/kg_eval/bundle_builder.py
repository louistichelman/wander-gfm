"""Build GraphBundle objects for transductive and inductive KG benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
from torch_geometric.data import Data

from ...graph_bundle import (
    GraphBundle,
    EdgeSetMode,
    InductiveFilterMode,
    Triple,
    append_inverse_edges,
    build_edge_set,
    build_kg_graph_object,
)
from .common import ensure_download, load_triple_file, merge_triple_lists


def build_transductive_bundle(
    *,
    name: str,
    train_triples: List[Triple],
    val_triples: List[Triple],
    test_triples: List[Triple],
    context_triples: Optional[List[Triple]] = None,
    num_nodes: int,
    num_forward_relations: int,
    add_inverse_edges_kgs: bool,
    edge_set_mode: EdgeSetMode = "train_only",
) -> GraphBundle:
    """Single-object transductive bundle (val/test alias train)."""
    ctx = context_triples if context_triples is not None else train_triples
    all_edges = merge_triple_lists(ctx, val_triples, test_triples)

    unique = list(dict.fromkeys(all_edges))
    edge_index = torch.tensor([[t[0], t[2]] for t in unique], dtype=torch.long).t()
    edge_type = torch.tensor([t[1] for t in unique], dtype=torch.long)
    t2i = {tr: i for i, tr in enumerate(unique)}

    n = edge_index.shape[1]
    visible = torch.zeros(n, dtype=torch.bool)
    train_m = torch.zeros(n, dtype=torch.bool)
    val_m = torch.zeros(n, dtype=torch.bool)
    test_m = torch.zeros(n, dtype=torch.bool)

    for tr in ctx:
        if tr in t2i:
            visible[t2i[tr]] = True
    for tr, m in ((train_triples, train_m), (val_triples, val_m), (test_triples, test_m)):
        for t in tr:
            if t in t2i:
                m[t2i[t]] = True

    data = Data(
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=num_nodes,
        num_relations=num_forward_relations,
    )
    data.visible_mask = visible
    data.train_mask = train_m
    data.val_mask = val_m
    data.test_mask = test_m
    data.is_link_prediction = True

    if add_inverse_edges_kgs:
        data = append_inverse_edges(data)
        n_fwd = n
        inv_false = torch.zeros(n_fwd, dtype=torch.bool)
        data.visible_mask = torch.cat([visible, visible])
        data.train_mask = torch.cat([train_m, inv_false])
        data.val_mask = torch.cat([val_m, inv_false])
        data.test_mask = torch.cat([test_m, inv_false])

    data.has_inverse_edges = bool(add_inverse_edges_kgs)
    data.edge_set = build_edge_set(data, edge_set_mode)

    return GraphBundle(
        train=data,
        val=None,
        test=None,
        name=name,
        is_knowledge_graph=True,
        is_inductive=False,
        inductive_filter_mode="transductive",
        edge_set_mode=edge_set_mode,
        has_predefined_edge_split=True,
    )


def build_undirected_transductive_bundle(
    *,
    name: str,
    train_triples: List[Triple],
    val_triples: List[Triple],
    test_triples: List[Triple],
    num_nodes: int,
    edge_set_mode: EdgeSetMode = "train_only",
) -> GraphBundle:
    """Single-relation **undirected** transductive LP bundle.

    Unlike :func:`build_transductive_bundle` (directed, optional KG inverse
    relations), this represents an undirected graph as a single edge type
    (``relation == 0``) with both directions stored explicitly in
    ``edge_index``:

    - ``visible_mask`` covers both directions of every train edge (context).
    - ``train_mask`` is True on both directions of train edges (queries train
      both predict-tail directions of each undirected edge).
    - ``val_mask`` / ``test_mask`` mark only the canonical (as-listed) direction
      of val/test edges; their reverses are present in ``edge_index`` but never
      visible and never queried.
    - ``inverse_edge_map`` matches ``(u, v) <-> (v, u)`` so the training loader
      can drop both directions of a sampled query edge from the context.

    ``has_inverse_edges`` is False and ``num_relations`` stays 1 (this does NOT
    use the KG new-relation-type mechanism). ``undirected_lp`` is set so the
    filtered-ranking helpers symmetrize their triple sets.
    """
    # Collect directed entries: every undirected edge -> both directions.
    order: List[Triple] = []  # directed (src, dst) pairs, relation implied 0
    index_of: dict = {}

    def _get_idx(s: int, d: int) -> int:
        key = (s, d)
        idx = index_of.get(key)
        if idx is None:
            idx = len(order)
            index_of[key] = idx
            order.append((s, d))
        return idx

    train_set = set()
    for h, _, t in train_triples:
        if h == t:
            continue
        _get_idx(h, t)
        _get_idx(t, h)
        train_set.add((h, t))
        train_set.add((t, h))

    # Register val/test entries (both directions exist for a symmetric graph /
    # valid inverse_edge_map) but only the canonical direction is queried, and
    # only if neither direction is a train edge (avoid leaking a visible answer).
    val_canon = []
    for h, _, t in val_triples:
        if h == t:
            continue
        _get_idx(h, t)
        _get_idx(t, h)
        if (h, t) not in train_set and (t, h) not in train_set:
            val_canon.append((h, t))
    test_canon = []
    for h, _, t in test_triples:
        if h == t:
            continue
        _get_idx(h, t)
        _get_idx(t, h)
        if (h, t) not in train_set and (t, h) not in train_set:
            test_canon.append((h, t))

    n = len(order)
    edge_index = torch.tensor([[s for s, _ in order], [d for _, d in order]], dtype=torch.long)
    edge_type = torch.zeros(n, dtype=torch.long)

    visible = torch.zeros(n, dtype=torch.bool)
    train_m = torch.zeros(n, dtype=torch.bool)
    val_m = torch.zeros(n, dtype=torch.bool)
    test_m = torch.zeros(n, dtype=torch.bool)

    for (s, d) in train_set:
        i = index_of[(s, d)]
        visible[i] = True
        train_m[i] = True
    for (h, t) in val_canon:
        val_m[index_of[(h, t)]] = True
    for (h, t) in test_canon:
        test_m[index_of[(h, t)]] = True

    inverse_edge_map = torch.tensor(
        [index_of[(d, s)] for (s, d) in order], dtype=torch.long
    )

    data = Data(
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=num_nodes,
        num_relations=1,
    )
    data.visible_mask = visible
    data.train_mask = train_m
    data.val_mask = val_m
    data.test_mask = test_m
    data.inverse_edge_map = inverse_edge_map
    data.is_link_prediction = True
    data.has_inverse_edges = False
    data.undirected_lp = True
    data.edge_set = build_edge_set(data, edge_set_mode)

    return GraphBundle(
        train=data,
        val=None,
        test=None,
        name=name,
        is_knowledge_graph=False,
        is_inductive=False,
        inductive_filter_mode="transductive",
        edge_set_mode=edge_set_mode,
        has_predefined_edge_split=True,
    )


def build_undirected_multirelation_bundle(
    *,
    name: str,
    train_triples: List[Triple],
    val_triples: List[Triple],
    test_triples: List[Triple],
    num_nodes: int,
    num_forward_relations: int,
    edge_set_mode: EdgeSetMode = "train_only",
) -> GraphBundle:
    """Multi-relation **undirected** transductive LP bundle.

    Generalizes :func:`build_undirected_transductive_bundle` to graphs with
    several symmetric relations (e.g. one relation per meta-path in the classic
    multiplex datasets). Directed entries are keyed by ``(src, dst, rel)``:

    - both directions of every edge are present in ``edge_index`` with the same
      ``edge_type``;
    - ``visible_mask`` / ``train_mask`` are True on both directions of train
      edges;
    - ``val_mask`` / ``test_mask`` mark only the canonical (as-listed) direction
      of val/test edges; their reverses exist in ``edge_index`` but are never
      visible and never queried;
    - ``inverse_edge_map`` matches ``(u, v, r) <-> (v, u, r)`` so the training
      loader can drop both directions of a sampled query edge.

    ``has_inverse_edges`` is False (relations are symmetric; no inverse-relation
    mechanism) and ``undirected_lp`` is set so the filtered-ranking helpers
    symmetrize their triple sets per relation.
    """
    order: List[Triple] = []  # directed (src, dst, rel) entries
    index_of: dict = {}

    def _get_idx(s: int, d: int, r: int) -> int:
        key = (s, d, r)
        idx = index_of.get(key)
        if idx is None:
            idx = len(order)
            index_of[key] = idx
            order.append((s, d, r))
        return idx

    train_set = set()
    for h, r, t in train_triples:
        if h == t:
            continue
        _get_idx(h, t, r)
        _get_idx(t, h, r)
        train_set.add((h, t, r))
        train_set.add((t, h, r))

    # Register val/test entries (both directions exist for a valid
    # inverse_edge_map) but only the canonical direction is queried, and only
    # if neither direction is a train edge of the same relation (avoid leaking
    # a visible answer).
    val_canon = []
    for h, r, t in val_triples:
        if h == t:
            continue
        _get_idx(h, t, r)
        _get_idx(t, h, r)
        if (h, t, r) not in train_set and (t, h, r) not in train_set:
            val_canon.append((h, t, r))
    test_canon = []
    for h, r, t in test_triples:
        if h == t:
            continue
        _get_idx(h, t, r)
        _get_idx(t, h, r)
        if (h, t, r) not in train_set and (t, h, r) not in train_set:
            test_canon.append((h, t, r))

    n = len(order)
    edge_index = torch.tensor(
        [[s for s, _, _ in order], [d for _, d, _ in order]], dtype=torch.long
    )
    edge_type = torch.tensor([r for _, _, r in order], dtype=torch.long)

    visible = torch.zeros(n, dtype=torch.bool)
    train_m = torch.zeros(n, dtype=torch.bool)
    val_m = torch.zeros(n, dtype=torch.bool)
    test_m = torch.zeros(n, dtype=torch.bool)

    for (s, d, r) in train_set:
        i = index_of[(s, d, r)]
        visible[i] = True
        train_m[i] = True
    for (h, t, r) in val_canon:
        val_m[index_of[(h, t, r)]] = True
    for (h, t, r) in test_canon:
        test_m[index_of[(h, t, r)]] = True

    inverse_edge_map = torch.tensor(
        [index_of[(d, s, r)] for (s, d, r) in order], dtype=torch.long
    )

    data = Data(
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=num_nodes,
        num_relations=num_forward_relations,
    )
    data.visible_mask = visible
    data.train_mask = train_m
    data.val_mask = val_m
    data.test_mask = test_m
    data.inverse_edge_map = inverse_edge_map
    data.is_link_prediction = True
    data.has_inverse_edges = False
    data.undirected_lp = True
    data.edge_set = build_edge_set(data, edge_set_mode)

    return GraphBundle(
        train=data,
        val=None,
        test=None,
        name=name,
        is_knowledge_graph=False,
        is_inductive=False,
        inductive_filter_mode="transductive",
        edge_set_mode=edge_set_mode,
        has_predefined_edge_split=True,
    )


def build_inductive_bundle(
    *,
    name: str,
    train_context: List[Triple],
    train_queries: List[Triple],
    val_context: List[Triple],
    val_queries: List[Triple],
    test_context: List[Triple],
    test_queries: List[Triple],
    train_num_nodes: int,
    val_num_nodes: int,
    test_num_nodes: int,
    train_num_forward_relations: int,
    val_num_forward_relations: int,
    test_num_forward_relations: int,
    add_inverse_edges_kgs: bool,
    inductive_filter_mode: InductiveFilterMode,
    edge_set_mode: EdgeSetMode = "train_only",
    dataset_version: Optional[str] = None,
) -> GraphBundle:
    train_obj = build_kg_graph_object(
        num_nodes=train_num_nodes,
        num_forward_relations=train_num_forward_relations,
        context_triples=train_context,
        query_triples=train_queries,
        active_query_split="train",
        add_inverse_edges=add_inverse_edges_kgs,
    )
    val_obj = build_kg_graph_object(
        num_nodes=val_num_nodes,
        num_forward_relations=val_num_forward_relations,
        context_triples=val_context,
        query_triples=val_queries,
        active_query_split="val",
        add_inverse_edges=add_inverse_edges_kgs,
    )
    test_obj = build_kg_graph_object(
        num_nodes=test_num_nodes,
        num_forward_relations=test_num_forward_relations,
        context_triples=test_context,
        query_triples=test_queries,
        active_query_split="test",
        add_inverse_edges=add_inverse_edges_kgs,
    )
    train_obj.edge_set = build_edge_set(train_obj, edge_set_mode)

    return GraphBundle(
        train=train_obj,
        val=val_obj,
        test=test_obj,
        name=name,
        is_knowledge_graph=True,
        is_inductive=True,
        inductive_filter_mode=inductive_filter_mode,
        edge_set_mode=edge_set_mode,
        has_predefined_edge_split=True,
        dataset_version=dataset_version,
    )


def load_three_split_transductive(
    data_dir: Path,
    name: str,
    urls: List[str],
    filenames: List[str],
    add_inverse_edges_kgs: bool,
    edge_set_mode: EdgeSetMode = "train_only",
    delimiter: Optional[str] = None,
    htr_order: bool = False,
    extra_train_context_files: Optional[List[str]] = None,
    raw_paths: Optional[List[Path]] = None,
) -> GraphBundle:
    if raw_paths is None:
        raw = data_dir / name / "raw"
        paths = []
        for url, fn in zip(urls, filenames):
            p = raw / fn
            ensure_download(url, p)
            paths.append(p)
    else:
        paths = list(raw_paths)

    raw = paths[0].parent

    train_t, n_ent, n_rel, ev, rv = load_triple_file(
        paths[0], delimiter=delimiter, htr_order=htr_order
    )
    val_t, n_ent, n_rel, ev, rv = load_triple_file(
        paths[1], delimiter=delimiter, htr_order=htr_order,
        inv_entity_vocab=ev, inv_rel_vocab=rv,
    )
    test_t, n_ent, n_rel, ev, rv = load_triple_file(
        paths[2], delimiter=delimiter, htr_order=htr_order,
        inv_entity_vocab=ev, inv_rel_vocab=rv,
    )

    ctx = list(train_t)
    if extra_train_context_files:
        for efn in extra_train_context_files:
            ep = raw / efn
            extra, n_ent, n_rel, ev, rv = load_triple_file(
                ep, delimiter=delimiter, htr_order=htr_order,
                inv_entity_vocab=ev, inv_rel_vocab=rv,
            )
            ctx = merge_triple_lists(ctx, extra)
            train_t = merge_triple_lists(train_t, extra)

    return build_transductive_bundle(
        name=name,
        train_triples=train_t,
        val_triples=val_t,
        test_triples=test_t,
        context_triples=ctx,
        num_nodes=n_ent,
        num_forward_relations=n_rel,
        add_inverse_edges_kgs=add_inverse_edges_kgs,
        edge_set_mode=edge_set_mode,
    )
