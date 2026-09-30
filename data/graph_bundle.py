"""GraphBundle: train/val/test split objects for transductive and inductive KGs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional, Set, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from .transforms import AddInverseEdges

SplitName = Literal["train", "val", "test"]
InductiveFilterMode = Literal["transductive", "grail", "ilpc", "hm_mtdea"]
EdgeSetMode = Literal["all_positives", "train_only"]

Triple = Tuple[int, int, int]  # (head, relation, tail)

# Bundle.metadata cache keys for filtered-ranking indexes (reused across epochs).
_EVAL_FILTER_CACHE_PREFIX = "_cached_eval_filter_index:"
_TRAIN_FILTER_CACHE_PREFIX = "_cached_train_filter_index:"


@dataclass(frozen=True)
class FilteredRankingIndex:
    """Vectorized filtered-ranking lookup: ``(h, r) -> tails`` / ``(t, r) -> heads``.

    Values are CPU ``int64`` tensors ready for ``scores[idx] = -inf``. Built once
    per dataset/split via NumPy sort+groupby and cached on ``bundle.metadata``.
    """

    tails: Dict[Tuple[int, int], torch.Tensor]  # (head, rel) -> filter tails
    heads: Dict[Tuple[int, int], torch.Tensor]  # (tail, rel) -> filter heads
    num_triples: int = 0



@dataclass
class GraphBundle:
    """Container for train/val/test graph objects.

    ``val`` and ``test`` may be ``None`` to alias the train object (transductive).
    """

    train: Optional[Data]
    val: Optional[Data] = None
    test: Optional[Data] = None
    name: str = ""
    is_knowledge_graph: bool = False
    is_inductive: bool = False
    inductive_filter_mode: InductiveFilterMode = "transductive"
    edge_set_mode: EdgeSetMode = "all_positives"
    has_node_features: bool = False
    has_node_labels: bool = False
    has_predefined_node_split: bool = False
    has_predefined_edge_split: bool = False
    dataset_version: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


def resolve_split(bundle: GraphBundle, split: SplitName) -> Data:
    """Return the graph object for ``split``; ``None`` slots alias train."""
    if split == "train":
        if bundle.train is None:
            raise ValueError(f"GraphBundle '{bundle.name}' has no train split.")
        return bundle.train
    if split == "val":
        return bundle.val if bundle.val is not None else resolve_split(bundle, "train")
    if split == "test":
        return bundle.test if bundle.test is not None else resolve_split(bundle, "train")
    raise ValueError(f"Unknown split: {split}")


def query_mask_for_split(data: Data, split: SplitName) -> torch.Tensor:
    """Return the query mask for ``split`` on a split-specific graph object."""
    attr = f"{split}_mask"
    mask = getattr(data, attr, None)
    if mask is None:
        raise AttributeError(
            f"Graph object has no {attr} for split '{split}' "
            f"(num_edges={data.edge_index.shape[1]})."
        )
    return mask


def is_link_prediction(data: Data) -> bool:
    """True when ``data`` represents link-prediction (edge-level query masks)."""
    if getattr(data, "is_link_prediction", None) is not None:
        return bool(data.is_link_prediction)
    if not hasattr(data, "visible_mask"):
        return False
    num_edges = data.edge_index.shape[1]
    for attr in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(data, attr, None)
        if mask is not None and len(mask) == num_edges:
            return True
    return False


def triples_from_masked_edges(data: Data, edge_mask: torch.Tensor) -> Set[Triple]:
    """Extract forward (h, r, t) triples for edges selected by ``edge_mask``.

    Prefer :func:`hrt_array_from_masked_edges` for large graphs; this set-based
    helper remains for training ``edge_set`` construction.
    """
    hrt = hrt_array_from_masked_edges(data, edge_mask)
    if hrt.size == 0:
        return set()
    return {(int(h), int(r), int(t)) for h, r, t in hrt}


def hrt_array_from_masked_edges(data: Data, edge_mask: torch.Tensor) -> np.ndarray:
    """Return ``int64`` array of shape ``[E, 3]`` with columns ``(h, r, t)``."""
    if edge_mask.numel() == 0 or int(edge_mask.sum().item()) == 0:
        return np.zeros((0, 3), dtype=np.int64)
    idx = edge_mask.nonzero(as_tuple=True)[0]
    heads = data.edge_index[0, idx].detach().cpu().numpy().astype(np.int64, copy=False)
    tails = data.edge_index[1, idx].detach().cpu().numpy().astype(np.int64, copy=False)
    rels = data.edge_type[idx].detach().cpu().numpy().astype(np.int64, copy=False)
    return np.stack([heads, rels, tails], axis=1)


def _forward_query_mask(data: Data, split: SplitName) -> torch.Tensor:
    """Query mask restricted to forward edges when inverses are present."""
    mask = query_mask_for_split(data, split)
    if getattr(data, "has_inverse_edges", False):
        n_forward = mask.shape[0] // 2
        return mask[:n_forward]
    return mask


def _combine_forward_edge_masks(data: Data, *masks: torch.Tensor) -> torch.Tensor:
    """OR boolean edge masks on forward edges only (inverse edges duplicate forward flags)."""
    if not masks:
        raise ValueError("At least one mask is required.")
    if getattr(data, "has_inverse_edges", False):
        n_forward = data.edge_index.shape[1] // 2
        masks = tuple(m[:n_forward] for m in masks)
    out = masks[0].clone()
    for mask in masks[1:]:
        out |= mask
    return out


def _with_inverse_triples(
    triples: Set[Triple], num_relations: int, has_inverse_edges: bool
) -> Set[Triple]:
    if not has_inverse_edges or not triples:
        return set(triples)
    num_forward = num_relations // 2
    out = set(triples)
    for h, r, t in triples:
        if r < num_forward:
            out.add((t, r + num_forward, h))
    return out


def _augment_hrt_array(
    hrt: np.ndarray,
    *,
    num_relations: int,
    has_inverse_edges: bool,
    undirected_lp: bool,
) -> np.ndarray:
    """Add inverse-relation and/or undirected reverse rows; return unique ``[E, 3]``."""
    if hrt.size == 0:
        return hrt.reshape(0, 3) if hrt.ndim != 2 else hrt
    parts: List[np.ndarray] = [hrt]
    if has_inverse_edges:
        num_forward = num_relations // 2
        fwd = hrt[hrt[:, 1] < num_forward]
        if fwd.size:
            inv = np.stack(
                [fwd[:, 2], fwd[:, 1] + num_forward, fwd[:, 0]],
                axis=1,
            )
            parts.append(inv)
    if undirected_lp:
        # Symmetrize every row currently in ``parts`` (including inverses if any).
        base = np.concatenate(parts, axis=0)
        rev = np.stack([base[:, 2], base[:, 1], base[:, 0]], axis=1)
        parts = [base, rev]
    out = np.concatenate(parts, axis=0)
    # unique rows; np.unique sorts which is fine for groupby later
    return np.unique(out, axis=0)


def _hrt_to_key_values_dict(hrt: np.ndarray) -> Dict[Tuple[int, int], torch.Tensor]:
    """Group ``[E, 3]=(key0, key1, value)`` rows into ``(key0, key1) -> LongTensor``."""
    if hrt.size == 0:
        return {}
    order = np.lexsort((hrt[:, 2], hrt[:, 1], hrt[:, 0]))
    hrt = hrt[order]
    change = np.empty(len(hrt), dtype=bool)
    change[0] = True
    change[1:] = (hrt[1:, 0] != hrt[:-1, 0]) | (hrt[1:, 1] != hrt[:-1, 1])
    starts = np.flatnonzero(change)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = len(hrt)
    out: Dict[Tuple[int, int], torch.Tensor] = {}
    for s, e in zip(starts.tolist(), ends.tolist()):
        key = (int(hrt[s, 0]), int(hrt[s, 1]))
        # from_numpy shares memory; copy so cached tensors stay valid if hrt is freed
        out[key] = torch.from_numpy(np.ascontiguousarray(hrt[s:e, 2], dtype=np.int64))
    return out


def filtered_ranking_index_from_hrt(hrt: np.ndarray) -> FilteredRankingIndex:
    """Build tail/head lookup dicts from unique ``[E, 3]`` ``(h, r, t)`` rows."""
    tails = _hrt_to_key_values_dict(hrt)
    # heads: group by (t, r) -> h  via column permutation
    if hrt.size == 0:
        heads: Dict[Tuple[int, int], torch.Tensor] = {}
    else:
        thr = np.stack([hrt[:, 2], hrt[:, 1], hrt[:, 0]], axis=1)
        heads = _hrt_to_key_values_dict(thr)
    return FilteredRankingIndex(tails=tails, heads=heads, num_triples=int(hrt.shape[0]))


def apply_filtered_entity_mask_(
    scores: torch.Tensor,
    filter_entities: Optional[torch.Tensor],
    true_entity: int,
) -> float:
    """In-place filtered ranking mask; returns the true entity's score.

    Sets all ``filter_entities`` to ``-inf``, then restores ``scores[true_entity]``
    so the target is never filtered out (matches the old set-minus semantics).
    """
    true_score = float(scores[true_entity].item())
    if filter_entities is not None and filter_entities.numel() > 0:
        scores[filter_entities] = float("-inf")
        scores[true_entity] = true_score
    return true_score


def _eval_filter_hrt_array(bundle: GraphBundle, split: SplitName) -> np.ndarray:
    """Collect filter triples as an HRT array for vectorized filtered ranking."""
    mode = bundle.inductive_filter_mode
    parts: List[np.ndarray] = []

    if mode == "transductive":
        data = resolve_split(bundle, "train")
        qmask = (
            _forward_query_mask(data, "train")
            | _forward_query_mask(data, "val")
            | _forward_query_mask(data, "test")
        )
        parts.append(hrt_array_from_masked_edges(data, qmask))
    elif mode == "grail" or mode == "hm_mtdea":
        data = resolve_split(bundle, split)
        edge_mask = _combine_forward_edge_masks(
            data, data.visible_mask, query_mask_for_split(data, split)
        )
        parts.append(hrt_array_from_masked_edges(data, edge_mask))
    elif mode == "ilpc":
        data = resolve_split(bundle, split)
        edge_mask = _combine_forward_edge_masks(
            data, data.visible_mask, query_mask_for_split(data, split)
        )
        parts.append(hrt_array_from_masked_edges(data, edge_mask))
        if split == "test":
            val_data = resolve_split(bundle, "val")
            val_edge_mask = _combine_forward_edge_masks(
                val_data, query_mask_for_split(val_data, "val")
            )
            parts.append(hrt_array_from_masked_edges(val_data, val_edge_mask))
    else:
        raise ValueError(f"Unknown inductive_filter_mode: {mode}")

    nonempty = [p for p in parts if p.size]
    if not nonempty:
        hrt = np.zeros((0, 3), dtype=np.int64)
    elif len(nonempty) == 1:
        hrt = nonempty[0]
    else:
        hrt = np.concatenate(nonempty, axis=0)

    ref_data = resolve_split(bundle, split)
    return _augment_hrt_array(
        hrt,
        num_relations=int(ref_data.num_relations),
        has_inverse_edges=bool(getattr(ref_data, "has_inverse_edges", False)),
        undirected_lp=bool(getattr(ref_data, "undirected_lp", False)),
    )


def _train_filter_hrt_array(bundle: GraphBundle, split: SplitName) -> np.ndarray:
    data = resolve_split(bundle, split)
    fwd_visible = _combine_forward_edge_masks(data, data.visible_mask)
    hrt = hrt_array_from_masked_edges(data, fwd_visible)
    return _augment_hrt_array(
        hrt,
        num_relations=int(data.num_relations),
        has_inverse_edges=bool(getattr(data, "has_inverse_edges", False)),
        undirected_lp=bool(getattr(data, "undirected_lp", False)),
    )


def build_eval_filter_index(bundle: GraphBundle, split: SplitName) -> FilteredRankingIndex:
    """Build a vectorized filtered-ranking index for full MRR/Hits eval."""
    return filtered_ranking_index_from_hrt(_eval_filter_hrt_array(bundle, split))


def build_train_filter_index(
    bundle: GraphBundle, split: SplitName = "test"
) -> FilteredRankingIndex:
    """Train-only filter index for the Recall@K protocol."""
    return filtered_ranking_index_from_hrt(_train_filter_hrt_array(bundle, split))


def get_eval_filter_index(bundle: GraphBundle, split: SplitName) -> FilteredRankingIndex:
    """Return a cached eval filter index on ``bundle.metadata`` (build once)."""
    key = f"{_EVAL_FILTER_CACHE_PREFIX}{split}"
    cached = bundle.metadata.get(key)
    if isinstance(cached, FilteredRankingIndex):
        return cached
    index = build_eval_filter_index(bundle, split)
    bundle.metadata[key] = index
    return index


def get_train_filter_index(
    bundle: GraphBundle, split: SplitName = "test"
) -> FilteredRankingIndex:
    """Return a cached train-only filter index on ``bundle.metadata``."""
    key = f"{_TRAIN_FILTER_CACHE_PREFIX}{split}"
    cached = bundle.metadata.get(key)
    if isinstance(cached, FilteredRankingIndex):
        return cached
    index = build_train_filter_index(bundle, split)
    bundle.metadata[key] = index
    return index


def build_edge_set(data: Data, mode: EdgeSetMode) -> Set[Triple]:
    """Build ``edge_set`` for negative-sampling false-negative filtering."""
    if mode == "all_positives":
        edge_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool)
    else:
        edge_mask = data.visible_mask.clone()
        if hasattr(data, "train_mask") and data.train_mask is not None:
            if len(data.train_mask) == data.edge_index.shape[1]:
                if getattr(data, "has_inverse_edges", False):
                    n_fwd = data.edge_index.shape[1] // 2
                    edge_mask[:n_fwd] |= data.train_mask[:n_fwd]
                else:
                    edge_mask |= data.train_mask
    triples = triples_from_masked_edges(data, edge_mask)
    return _with_inverse_triples(
        triples,
        int(data.num_relations),
        bool(getattr(data, "has_inverse_edges", False)),
    )


def append_inverse_edges(data: Data) -> Data:
    """Append explicit inverse edges; sets ``has_inverse_edges``."""
    data = AddInverseEdges()(data)
    data.has_inverse_edges = True
    return data


def _dedupe_triples(
    triples: Iterable[Triple],
) -> Tuple[List[Triple], Dict[Triple, int]]:
    """Return unique triples and index map."""
    seen: Dict[Triple, int] = {}
    ordered: List[Triple] = []
    for tr in triples:
        if tr not in seen:
            seen[tr] = len(ordered)
            ordered.append(tr)
    return ordered, seen


def build_kg_graph_object(
    *,
    num_nodes: int,
    num_forward_relations: int,
    context_triples: Iterable[Triple],
    query_triples: Iterable[Triple],
    active_query_split: SplitName,
    add_inverse_edges: bool,
) -> Data:
    """Build one KG ``Data`` from context + query triples (unified edge_index)."""
    ctx_list = list(context_triples)
    ctx_set = set(ctx_list)
    q_list = list(query_triples)

    all_triples: List[Triple] = list(ctx_list)
    for tr in q_list:
        if tr not in ctx_set:
            all_triples.append(tr)
            ctx_set.add(tr)

    if not all_triples:
        raise ValueError("Cannot build KG graph object with zero triples.")

    unique, _ = _dedupe_triples(all_triples)
    edge_index = torch.tensor(
        [[t[0], t[2]] for t in unique], dtype=torch.long
    ).t()
    edge_type = torch.tensor([t[1] for t in unique], dtype=torch.long)

    n_edges = edge_index.shape[1]
    visible_mask = torch.zeros(n_edges, dtype=torch.bool)
    query_mask = torch.zeros(n_edges, dtype=torch.bool)

    triple_to_idx = {tr: i for i, tr in enumerate(unique)}
    for tr in ctx_list:
        if tr in triple_to_idx:
            visible_mask[triple_to_idx[tr]] = True
    for tr in q_list:
        if tr in triple_to_idx:
            query_mask[triple_to_idx[tr]] = True
            # Query triples not in context stay hidden at eval (Wander-aligned).
            # Overlapping query/context edges are already marked visible above.

    data = Data(
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=num_nodes,
        num_relations=num_forward_relations,
    )
    data.visible_mask = visible_mask
    data.train_mask = None
    data.val_mask = None
    data.test_mask = None
    setattr(data, f"{active_query_split}_mask", query_mask)

    if add_inverse_edges:
        data = append_inverse_edges(data)
        n_fwd = n_edges
        pre_vis = visible_mask
        data.visible_mask = torch.cat([pre_vis, pre_vis])
        inv_false = torch.zeros(n_fwd, dtype=torch.bool)
        setattr(
            data,
            f"{active_query_split}_mask",
            torch.cat([query_mask, inv_false]),
        )
    data.is_link_prediction = True
    data.has_inverse_edges = bool(add_inverse_edges)
    return data


def wrap_transductive_data(
    data: Data,
    *,
    name: str,
    edge_set_mode: EdgeSetMode,
    has_node_features: bool,
    has_node_labels: bool,
    has_predefined_node_split: bool,
    has_predefined_edge_split: bool,
) -> GraphBundle:
    """Wrap a processed transductive ``Data`` object into a ``GraphBundle``."""
    is_kg = is_link_prediction(data)
    if is_kg:
        data.edge_set = build_edge_set(data, edge_set_mode)
    bundle = GraphBundle(
        train=data,
        val=None,
        test=None,
        name=name,
        is_knowledge_graph=is_kg,
        is_inductive=False,
        inductive_filter_mode="transductive",
        edge_set_mode=edge_set_mode,
        has_node_features=has_node_features,
        has_node_labels=has_node_labels,
        has_predefined_node_split=has_predefined_node_split,
        has_predefined_edge_split=has_predefined_edge_split,
    )
    return bundle


def bundle_to_legacy_data(bundle: GraphBundle) -> Data:
    """Return train object for backward-compatible single-``Data`` consumers."""
    if bundle.is_inductive:
        raise ValueError(
            f"Cannot convert inductive bundle '{bundle.name}' to legacy Data; "
            "use resolve_split(bundle, split) instead."
        )
    return resolve_split(bundle, "train")
