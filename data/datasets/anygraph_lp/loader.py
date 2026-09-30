"""Convert AnyGraph pickle adjacency matrices into transductive KG bundles.

Each AnyGraph link-prediction dataset is treated as a single-relation
(``relation == 0``) transductive knowledge graph. Homogeneous datasets keep
node ids ``0..N-1``; bipartite (user x item) datasets concatenate the node
spaces as ``[users | items]`` with items offset by ``user_num`` (matching
AnyGraph, which ranks only item nodes at test time).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import scipy.sparse as sp
import torch

from ...graph_bundle import GraphBundle, Triple
from ..kg_eval.bundle_builder import (
    build_transductive_bundle,
    build_undirected_transductive_bundle,
)
from .common import ensure_anygraph_raw


def _load_coo(path: Path) -> sp.coo_matrix:
    """Load a pickled (sparse/dense) matrix, binarize it, and return COO."""
    with open(path, "rb") as fh:
        mat = pickle.load(fh)
    mat = mat != 0
    if not sp.issparse(mat):
        mat = sp.coo_matrix(mat)
    return mat.tocoo()


def _triples_from_coo(mat: sp.coo_matrix, tail_offset: int) -> List[Triple]:
    """Build forward ``(head, 0, tail + offset)`` triples from a COO matrix."""
    rows = mat.row.tolist()
    cols = mat.col.tolist()
    return [(int(r), 0, int(c) + tail_offset) for r, c in zip(rows, cols)]


def _train_out_neighbors_from_coo(
    mat: sp.coo_matrix, tail_offset: int
) -> Dict[int, List[int]]:
    """Outgoing train neighbors per source (AnyGraph ``csr_mat[row]`` semantics)."""
    out: Dict[int, List[int]] = {}
    for r, c in zip(mat.row.tolist(), mat.col.tolist()):
        h, t = int(r), int(c) + tail_offset
        out.setdefault(h, []).append(t)
    return out


def load_anygraph_bundle(
    data_dir: str,
    folder: str,
    *,
    display_name: str,
    add_inverse_edges_kgs: bool = True,
) -> GraphBundle:
    """Load an AnyGraph link-prediction dataset as a transductive KG bundle.

    Args:
        data_dir: Root dataset directory (``--data_dir``).
        folder: AnyGraph dataset folder name inside the archive (e.g. ``cora``).
        display_name: Bundle / registry display name (e.g. ``ANYGRAPH_CORA``).
        add_inverse_edges_kgs: If True, append explicit inverse edges
            (relation ``1``). When False, the single forward relation is used
            and Wander handles directionality. Only honored for bipartite
            datasets; non-bipartite graphs are loaded as undirected
            single-relation LP and ignore this flag.
    """
    ds_dir = ensure_anygraph_raw(Path(data_dir), folder)

    trn = _load_coo(ds_dir / "trn_mat.pkl")
    val = _load_coo(ds_dir / "val_mat.pkl")
    tst = _load_coo(ds_dir / "tst_mat.pkl")

    is_bipartite = trn.shape[0] != trn.shape[1]
    if is_bipartite:
        user_num, item_num = trn.shape
        num_nodes = user_num + item_num
        tail_offset = user_num
    else:
        num_nodes = int(trn.shape[0])
        tail_offset = 0

    if is_bipartite:
        # Directed user -> item recommendation graph (ranks item nodes only).
        bundle = build_transductive_bundle(
            name=display_name,
            train_triples=_triples_from_coo(trn, tail_offset),
            val_triples=_triples_from_coo(val, tail_offset),
            test_triples=_triples_from_coo(tst, tail_offset),
            num_nodes=num_nodes,
            num_forward_relations=1,
            add_inverse_edges_kgs=add_inverse_edges_kgs,
            edge_set_mode="train_only",
        )
    else:
        # Homogeneous graph: undirected single-relation link prediction.
        bundle = build_undirected_transductive_bundle(
            name=display_name,
            train_triples=_triples_from_coo(trn, tail_offset),
            val_triples=_triples_from_coo(val, tail_offset),
            test_triples=_triples_from_coo(tst, tail_offset),
            num_nodes=num_nodes,
            edge_set_mode="train_only",
        )

    data = bundle.train
    # Candidate restriction metadata used by the Recall@K protocol: bipartite
    # datasets rank only item nodes in range [candidate_offset, num_nodes).
    # NB: avoid the name ``is_bipartite`` -- PyG's ``Data`` exposes a method of
    # that name, which would shadow a stored attribute on getattr access.
    data.anygraph_bipartite = bool(is_bipartite)
    data.anygraph_candidate_offset = int(tail_offset)
    # Directed train out-neighbors (NBFNet / AnyGraph CSR-row semantics).
    data.anygraph_train_out_neighbors = _train_out_neighbors_from_coo(trn, tail_offset)

    feat_path = ds_dir / "feats.pkl"
    if feat_path.exists():
        with open(feat_path, "rb") as fh:
            feats = pickle.load(fh)
        x = torch.from_numpy(np.asarray(feats)).float()
        if x.ndim == 2 and x.shape[0] == num_nodes:
            data.x = x
            bundle.has_node_features = True
        else:
            print(
                f"[anygraph_lp] {display_name}: skipping feats.pkl with shape "
                f"{tuple(x.shape)} (expected [{num_nodes}, d]); running featureless."
            )

    return bundle
