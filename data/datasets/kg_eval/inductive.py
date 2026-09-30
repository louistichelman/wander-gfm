"""Inductive KG bundle loaders (GraIL, ILPC, InGram, HM, MTDEA)."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import List, Optional

import torch

from ...graph_bundle import GraphBundle, Triple
from .bundle_builder import build_inductive_bundle
from .common import ensure_download, load_triple_file, merge_triple_lists


def _download_grail_files(raw: Path, prefix: str, version: str) -> None:
    base = f"https://raw.githubusercontent.com/kkteru/grail/master/data/{prefix}_{version}"
    ind_base = f"https://raw.githubusercontent.com/kkteru/grail/master/data/{prefix}_{version}_ind"
    mapping = [
        (f"{ind_base}/train.txt", "train_ind.txt"),
        (f"{ind_base}/valid.txt", "valid_ind.txt"),
        (f"{ind_base}/test.txt", "test_ind.txt"),
        (f"{base}/train.txt", "train.txt"),
        (f"{base}/valid.txt", "valid.txt"),
    ]
    for url, fn in mapping:
        ensure_download(url, raw / fn)


def load_grail_bundle(
    data_dir: Path,
    name: str,
    prefix: str,
    version: str,
    add_inverse_edges_kgs: bool,
    merge_valid_test: bool = True,
) -> GraphBundle:
    raw = data_dir / name / version / "raw"
    _download_grail_files(raw, prefix, version)

    train_ev: dict = {}
    train_rv: dict = {}
    test_ev: dict = {}
    test_rv: dict = {}

    train_ctx, n_train, n_rel, train_ev, train_rv = load_triple_file(raw / "train.txt")
    train_q = list(train_ctx)
    val_q, n_train, n_rel, train_ev, train_rv = load_triple_file(
        raw / "valid.txt", inv_entity_vocab=train_ev, inv_rel_vocab=train_rv
    )

    test_ctx, _, _, test_ev, test_rv = load_triple_file(raw / "train_ind.txt")
    valid_ind, _, _, test_ev, test_rv = load_triple_file(
        raw / "valid_ind.txt", inv_entity_vocab=test_ev, inv_rel_vocab=test_rv
    )
    # n_test includes query-only entities from valid_ind / test_ind.
    test_ind, n_test, n_rel_test, test_ev, test_rv = load_triple_file(
        raw / "test_ind.txt", inv_entity_vocab=test_ev, inv_rel_vocab=test_rv
    )
    test_q = merge_triple_lists(valid_ind, test_ind) if merge_valid_test else test_ind

    return build_inductive_bundle(
        name=name,
        train_context=train_ctx,
        train_queries=train_q,
        val_context=train_ctx,
        val_queries=val_q,
        test_context=test_ctx,
        test_queries=test_q,
        train_num_nodes=n_train,
        val_num_nodes=n_train,
        test_num_nodes=n_test,
        train_num_forward_relations=n_rel,
        val_num_forward_relations=n_rel,
        test_num_forward_relations=n_rel,
        add_inverse_edges_kgs=add_inverse_edges_kgs,
        inductive_filter_mode="grail",
        dataset_version=version,
    )


def _download_four_file_inductive(
    raw: Path, urls: List[str], filenames: List[str], version: str
) -> None:
    for url, fn in zip(urls, filenames):
        if url == "local":
            if not (raw / fn).exists():
                raise FileNotFoundError(f"Missing local inductive file: {raw / fn}")
            continue
        ensure_download(url % version, raw / fn)


def load_generic_inductive_bundle(
    data_dir: Path,
    name: str,
    urls: List[str],
    filenames: List[str],
    version: str,
    add_inverse_edges_kgs: bool,
    valid_on_inf: bool,
    filter_mode: str,
) -> GraphBundle:
    raw = data_dir / name / version / "raw"
    _download_four_file_inductive(raw, urls, filenames, version)

    train_t, n_train, n_train_rel, train_ev, train_rv = load_triple_file(
        raw / filenames[0]
    )
    inf_t, _, _, inf_ev, inf_rv = load_triple_file(raw / filenames[1])

    if valid_on_inf:
        # Val and test share the inference vocab, so grow it once (val then test)
        # and size both graphs to the union. Isolated query-only entities become
        # extra nodes so filtered ranking has a score slot for them.
        val_t, _, _, inf_ev, inf_rv = load_triple_file(
            raw / filenames[2], inv_entity_vocab=inf_ev, inv_rel_vocab=inf_rv
        )
        test_t, test_n, test_rel, inf_ev, inf_rv = load_triple_file(
            raw / filenames[3], inv_entity_vocab=inf_ev, inv_rel_vocab=inf_rv
        )
        val_num_nodes, test_num_nodes = test_n, test_n
        val_num_rel, test_num_rel = test_rel, test_rel
    else:
        val_t, val_n, val_rel, _, _ = load_triple_file(
            raw / filenames[2], inv_entity_vocab=train_ev, inv_rel_vocab=train_rv
        )
        test_t, test_n, test_rel, _, _ = load_triple_file(
            raw / filenames[3], inv_entity_vocab=inf_ev, inv_rel_vocab=inf_rv
        )
        val_num_nodes, test_num_nodes = val_n, test_n
        val_num_rel, test_num_rel = val_rel, test_rel

    return build_inductive_bundle(
        name=name,
        train_context=train_t,
        train_queries=train_t,
        val_context=inf_t if valid_on_inf else train_t,
        val_queries=val_t,
        test_context=inf_t,
        test_queries=test_t,
        train_num_nodes=n_train,
        val_num_nodes=val_num_nodes,
        test_num_nodes=test_num_nodes,
        train_num_forward_relations=n_train_rel,
        val_num_forward_relations=val_num_rel,
        test_num_forward_relations=test_num_rel,
        add_inverse_edges_kgs=add_inverse_edges_kgs,
        inductive_filter_mode=filter_mode,  # type: ignore[arg-type]
        dataset_version=version,
    )


def load_hm_bundle(
    data_dir: Path,
    name: str,
    version: str,
    folder: str,
    add_inverse_edges_kgs: bool,
) -> GraphBundle:
    raw = data_dir / name / version / "raw"
    base_url = f"https://raw.githubusercontent.com/shuwen-liu-ox/INDIGO/master/data/{folder}"
    for rel, fn in [
        ("train/train.txt", "train.txt"),
        ("train/valid.txt", "valid.txt"),
        ("test/test-graph.txt", "test-graph.txt"),
        ("test/test-fact.txt", "test-fact.txt"),
    ]:
        ensure_download(f"{base_url}/{rel}", raw / fn)

    train_t, n_train, n_rel, ev, rv = load_triple_file(raw / "train.txt")
    val_t, val_n, _, ev, rv = load_triple_file(
        raw / "valid.txt", inv_entity_vocab=ev, inv_rel_vocab=rv
    )
    test_ctx, _, _, tev, trv = load_triple_file(raw / "test-graph.txt")
    # Query-only entities (degree 0 in test-graph) still need a node slot.
    test_q, n_test, n_test_rel, tev, trv = load_triple_file(
        raw / "test-fact.txt", inv_entity_vocab=tev, inv_rel_vocab=trv
    )

    return build_inductive_bundle(
        name=name,
        train_context=train_t,
        train_queries=train_t,
        val_context=train_t,
        val_queries=val_t,
        test_context=test_ctx,
        test_queries=test_q,
        train_num_nodes=n_train,
        val_num_nodes=val_n,
        test_num_nodes=n_test,
        train_num_forward_relations=n_rel,
        val_num_forward_relations=n_rel,
        test_num_forward_relations=n_test_rel,
        add_inverse_edges_kgs=add_inverse_edges_kgs,
        inductive_filter_mode="hm_mtdea",
        dataset_version=version,
    )
