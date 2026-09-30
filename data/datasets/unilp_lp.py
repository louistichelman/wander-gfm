"""UniLP (context_LP) homogeneous LP graphs and 70/10/20 edge splits.

Copied from ``context_LP/utils.py`` without pulling ogb / heuristics.
Cached split files use the UniLP name ``split{run}_{val}_{test}.pt`` so existing
UniLP caches can be reused as-is.

``.mat`` graphs live under ``{data_dir}/UniLP/`` (copied from
``third_party/context_LP/data``). Cora is loaded via PyG ``Planetoid``. CS via
``Coauthor``. Facebook via ``snap_dataset.SNAPDataset`` from the context_LP
checkout (``snap-musae-facebook``).
"""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.data.collate import collate
from torch_geometric.datasets import Coauthor, Planetoid
from torch_geometric.utils import (
    add_self_loops,
    from_scipy_sparse_matrix,
    is_undirected,
    negative_sampling,
    to_undirected,
    train_test_split_edges,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes

# Paper Table 1 test graphs (minus CS / Facebook, which are larger).
UNILP_SMALL_DATASETS: List[str] = ["Celegans", "USAir", "NS", "PB", "Cora"]
UNILP_LARGE_DATASETS: List[str] = ["CS", "snap-musae-facebook"]
UNILP_TABLE1_DATASETS: List[str] = UNILP_SMALL_DATASETS + UNILP_LARGE_DATASETS
UNILP_DIRNAME = "UniLP"
_COAUTHOR_PYG = {"cs": "CS"}
_SNAP_NAMES = {"snap-musae-facebook": "musae-facebook"}

_MAT_DATASETS = frozenset({
    "Celegans",
    "USAir",
    "NS",
    "PB",
})

_PLANETOID_FOLDERS = ("Cora",)
_PLANETOID_PYG = {name.lower(): name for name in _PLANETOID_FOLDERS}
_FEATURED_DATASETS = (
    frozenset(_PLANETOID_FOLDERS)
    | frozenset(_COAUTHOR_PYG.values())
    | frozenset(_SNAP_NAMES)
)

# Registry keys for every graph this loader can currently materialize.
UNILP_ALL_DATASETS: List[str] = sorted(
    set(_MAT_DATASETS) | set(_PLANETOID_FOLDERS) | set(_COAUTHOR_PYG.values()) | set(_SNAP_NAMES)
)

_UNILP_SPLIT_ALIASES = {
    "unilp_small": UNILP_SMALL_DATASETS,
    "unilp": UNILP_SMALL_DATASETS,
    "unilp_table1": UNILP_TABLE1_DATASETS,
    "unilp_large": UNILP_LARGE_DATASETS,
}


def removerepeated(edge_index: torch.Tensor) -> torch.Tensor:
    """Undirected upper-triangle unique pairs ``(u, v)`` with ``u < v``."""
    edge_index = to_undirected(edge_index)
    return edge_index[:, edge_index[0] < edge_index[1]]


def randomsplit(
    data: Data,
    val_ratio: float = 0.10,
    test_ratio: float = 0.2,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """UniLP ``randomsplit``: PyG ``train_test_split_edges`` then re-slice val.

    PyG is called with ``val_ratio=test_ratio`` and ``test_ratio=test_ratio``.
    Half of that PyG val slice is moved back to train so the cached protocol is
    ~70/10/20 (exactly ``val_ratio`` / ``test_ratio`` of the unique undirected
    edges, matching ``context_LP/utils.py``).
    """
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*train_test_split_edges.*",
            category=UserWarning,
        )
        data = train_test_split_edges(data, test_ratio, test_ratio)
    split_edge: Dict[str, Dict[str, torch.Tensor]] = {
        "train": {},
        "valid": {},
        "test": {},
    }
    num_val = int(data.val_pos_edge_index.shape[1] * val_ratio / test_ratio)
    data.val_pos_edge_index = data.val_pos_edge_index[
        :, torch.randperm(data.val_pos_edge_index.shape[1])
    ]
    split_edge["train"]["edge"] = removerepeated(
        torch.cat(
            (data.train_pos_edge_index, data.val_pos_edge_index[:, :-num_val]),
            dim=-1,
        )
    ).t()
    split_edge["valid"]["edge"] = removerepeated(
        data.val_pos_edge_index[:, -num_val:]
    ).t()
    split_edge["valid"]["edge_neg"] = removerepeated(data.val_neg_edge_index).t()
    split_edge["test"]["edge"] = removerepeated(data.test_pos_edge_index).t()
    split_edge["test"]["edge_neg"] = removerepeated(data.test_neg_edge_index).t()

    edge_index = to_undirected(split_edge["train"]["edge"].t())
    new_edge_index, _ = add_self_loops(edge_index)
    neg_edge = negative_sampling(
        new_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=split_edge["train"]["edge"].size(0),
    )
    split_edge["train"]["edge_neg"] = neg_edge.t()
    return split_edge


def _load_split_file(path: Path) -> Dict[str, Dict[str, torch.Tensor]]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def edge_split(
    data: Data,
    root: Path,
    name: str,
    split_val_percent: int = 10,
    split_test_percent: int = 20,
    run: int = 0,
) -> Tuple[Data, Dict[str, Dict[str, torch.Tensor]]]:
    """Load or create ``split{run}_{val}_{test}.pt`` and set train-only edges."""
    val_ratio = split_val_percent / 100
    test_ratio = split_test_percent / 100
    # Capture N before ``train_test_split_edges`` nulls ``edge_index``.
    num_nodes = int(maybe_num_nodes(data.edge_index, data.num_nodes))
    data_folder = Path(root) / name
    data_folder.mkdir(parents=True, exist_ok=True)
    file_path = data_folder / f"split{run}_{split_val_percent}_{split_test_percent}.pt"
    if file_path.exists():
        print(f"load split edge from {file_path}")
        split_edge = _load_split_file(file_path)
    else:
        require = os.environ.get("UNILP_REQUIRE_EXISTING_SPLITS", "").strip().lower()
        if require in {"1", "true", "yes"}:
            raise FileNotFoundError(
                f"UniLP split missing: {file_path}. "
                "Refusing to generate a new split (UNILP_REQUIRE_EXISTING_SPLITS=1)."
            )
        split_edge = randomsplit(data, val_ratio=val_ratio, test_ratio=test_ratio)
        torch.save(split_edge, file_path)
        print(f"save split edges to {file_path}")
    data.edge_index = to_undirected(
        split_edge["train"]["edge"].t(), num_nodes=num_nodes
    )
    data.num_nodes = num_nodes
    return data, split_edge


def _context_lp_root() -> Path:
    from baselines.paths import resolve_context_lp_root

    return resolve_context_lp_root()


def _load_snap_graph(snap_name: str, dataset_dir: Path) -> Data:
    """Load a MUSAE SNAP graph via context_LP's SNAPDataset."""
    import sys

    unilp_root = _context_lp_root()
    if str(unilp_root) not in sys.path:
        sys.path.insert(0, str(unilp_root))
    from snap_dataset import SNAPDataset  # noqa: E402

    dataset = SNAPDataset(str(dataset_dir), snap_name)
    data, _, _ = collate(
        dataset[0].__class__,
        data_list=list(dataset),
        increment=True,
        add_batch=False,
    )
    if data.edge_index is not None and not is_undirected(data.edge_index):
        data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    return data


def _load_mat_graph(name: str, dataset_dir: Path) -> Data:
    import scipy.io as sio

    mat_path = dataset_dir / f"{name}.mat"
    if not mat_path.is_file():
        raise FileNotFoundError(
            f"UniLP .mat graph not found: {mat_path}. "
            "Copy third_party/context_LP/data/*.mat to {data_dir}/UniLP/ "
            "(Celegans.mat, USAir.mat, NS.mat, PB.mat, ...)."
        )
    print(f"Load data from: {mat_path}")
    net = sio.loadmat(str(mat_path))
    edge_index, _ = from_scipy_sparse_matrix(net["net"])
    data = Data(
        edge_index=edge_index,
        num_nodes=int(torch.max(edge_index).item()) + 1,
    )
    if not is_undirected(data.edge_index):
        data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    return data


def _load_raw_graph(dataset_name: str, dataset_dir: Path) -> Data:
    if dataset_name.lower() in _PLANETOID_PYG:
        dataset = Planetoid(str(dataset_dir), _PLANETOID_PYG[dataset_name.lower()])
        data, _, _ = collate(
            dataset[0].__class__,
            data_list=list(dataset),
            increment=True,
            add_batch=False,
        )
        return data
    if dataset_name in _MAT_DATASETS:
        data = _load_mat_graph(dataset_name, dataset_dir)
        data, _, _ = collate(
            data.__class__,
            data_list=[data],
            increment=True,
            add_batch=False,
        )
        return data
    pyg_name = _COAUTHOR_PYG.get(dataset_name.lower())
    if pyg_name is not None:
        dataset = Coauthor(str(dataset_dir), pyg_name)
        data, _, _ = collate(
            dataset[0].__class__,
            data_list=list(dataset),
            increment=True,
            add_batch=False,
        )
        return data
    snap_name = _SNAP_NAMES.get(dataset_name)
    if snap_name is not None:
        return _load_snap_graph(snap_name, dataset_dir)
    raise ValueError(
        f"Unsupported UniLP dataset {dataset_name!r}. "
        f"Implemented: {UNILP_ALL_DATASETS}."
    )


def load_unilp_split(
    dataset_name: str,
    data_dir: str | Path,
    run: int = 0,
    split_val_percent: int = 10,
    split_test_percent: int = 20,
) -> Tuple[Data, Dict[str, Dict[str, torch.Tensor]]]:
    """Load a UniLP graph and its 70/10/20 pos/neg split.

    After this call, ``data.edge_index`` is **train positives only** (undirected).
    Val edges are not added to the graph.
    """
    dataset_dir = Path(data_dir)
    print(f"Loading dataset {dataset_name}")
    data = _load_raw_graph(dataset_name, dataset_dir)
    data, split_edge = edge_split(
        data,
        dataset_dir,
        dataset_name,
        split_val_percent=split_val_percent,
        split_test_percent=split_test_percent,
        run=run,
    )
    data.num_nodes = int(maybe_num_nodes(data.edge_index, data.num_nodes))
    return data, split_edge


def to_wander_lp_data(data: Data, *, ignore_features: bool = True) -> Data:
    """Minimal homogeneous LP ``Data`` for Wander (train-only edges, rel=0)."""
    num_nodes = int(maybe_num_nodes(data.edge_index, data.num_nodes))
    edge_index = to_undirected(data.edge_index, num_nodes=num_nodes)
    out = Data(
        edge_index=edge_index,
        edge_type=torch.zeros(edge_index.size(1), dtype=torch.long),
        num_nodes=num_nodes,
        num_relations=1,
        is_link_prediction=True,
    )
    if not ignore_features and getattr(data, "x", None) is not None:
        out.x = data.x
    return out


def resolve_unilp_data_dir(data_dir: str | Path) -> Path:
    """Return the folder that contains UniLP ``*.mat`` files.

    Prefers ``{data_dir}/UniLP/``; falls back to ``data_dir`` itself when the
    mats were placed there directly (e.g. ``third_party/context_LP/data``).
    """
    root = Path(data_dir)
    nested = root / UNILP_DIRNAME
    if any(nested.glob("*.mat")):
        return nested
    if any(root.glob("*.mat")):
        return root
    return nested


def _pairs_to_e2(pairs: Optional[torch.Tensor]) -> torch.Tensor:
    if pairs is None:
        return torch.empty((0, 2), dtype=torch.long)
    t = pairs.detach().cpu().long()
    if t.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long)
    return t.reshape(-1, 2)


def _pairs_to_triples(pairs: Optional[torch.Tensor]) -> List[Tuple[int, int, int]]:
    return [(int(h), 0, int(t)) for h, t in _pairs_to_e2(pairs).tolist()]


def load_unilp_bundle(
    data_dir: str | Path,
    dataset_name: str,
    *,
    display_name: str,
    seed: int = 0,
    add_inverse_edges_kgs: bool = True,
    dataset_version: Optional[str] = None,
):
    """Load a UniLP graph as an undirected transductive LP ``GraphBundle``.

    Official pos/neg pairs for the Hits/AUC protocol are stored on ``bundle.train``
    as ``unilp_{val,test}_{pos,neg}`` (``[E, 2]``). ``seed`` is the UniLP split
    ``run`` index (cached as ``split{run}_10_20.pt``).
    """
    del add_inverse_edges_kgs, dataset_version
    from data.datasets.kg_eval.bundle_builder import build_undirected_transductive_bundle

    unilp_dir = resolve_unilp_data_dir(data_dir)
    pyg_data, split_edge = load_unilp_split(
        dataset_name,
        unilp_dir,
        run=int(seed),
    )
    train_pairs = _pairs_to_e2(split_edge["train"]["edge"])
    val_pairs = _pairs_to_e2(split_edge["valid"]["edge"])
    test_pairs = _pairs_to_e2(split_edge["test"]["edge"])
    val_neg = _pairs_to_e2(split_edge["valid"].get("edge_neg"))
    test_neg = _pairs_to_e2(split_edge["test"].get("edge_neg"))
    train_neg = _pairs_to_e2(split_edge["train"].get("edge_neg"))

    num_nodes = int(maybe_num_nodes(pyg_data.edge_index, pyg_data.num_nodes))
    bundle = build_undirected_transductive_bundle(
        name=display_name,
        train_triples=_pairs_to_triples(train_pairs),
        val_triples=_pairs_to_triples(val_pairs),
        test_triples=_pairs_to_triples(test_pairs),
        num_nodes=num_nodes,
        edge_set_mode="train_only",
    )
    data = bundle.train
    data.unilp_train_pos = train_pairs
    data.unilp_train_neg = train_neg
    data.unilp_val_pos = val_pairs
    data.unilp_val_neg = val_neg
    data.unilp_test_pos = test_pairs
    data.unilp_test_neg = test_neg
    data.unilp_folder = dataset_name
    data.unilp_data_dir = str(unilp_dir)
    data.unilp_run = int(seed)
    data.unilp_split_edge = split_edge
    if getattr(pyg_data, "x", None) is not None:
        data.x = pyg_data.x
        bundle.has_node_features = True
    return bundle


def _registry_key(name: str) -> str:
    return "UNILP_" + name.upper().replace("-", "_").replace(".", "_")


def unilp_small_registry_keys() -> List[str]:
    return [_registry_key(name) for name in UNILP_SMALL_DATASETS]


def expand_unilp_dataset_aliases(names: List[str]) -> List[str]:
    """Expand ``unilp`` / ``unilp_small`` placeholders to ``UNILP_*`` keys."""
    expanded: List[str] = []
    for name in names:
        folders = _UNILP_SPLIT_ALIASES.get(name.lower())
        if folders is not None:
            expanded.extend(_registry_key(folder) for folder in folders)
        else:
            expanded.append(name)
    return expanded


def _make_module(folder: str) -> ModuleType:
    key = _registry_key(folder)

    def loader(
        data_dir: str,
        seed: int = 42,
        add_inverse_edges_kgs: bool = True,
        dataset_version: Optional[str] = None,
    ):
        return load_unilp_bundle(
            data_dir,
            folder,
            display_name=key,
            seed=seed,
            add_inverse_edges_kgs=add_inverse_edges_kgs,
            dataset_version=dataset_version,
        )

    m = ModuleType(key)
    m.NAME = key
    m.load_graph_bundle = loader
    m.load_base_dataset = None  # type: ignore[assignment]
    m.HAS_NODE_FEATURES = folder in _FEATURED_DATASETS
    m.HAS_NODE_LABELS = False
    m.HAS_PREDEFINED_NODE_SPLIT = False
    m.HAS_PREDEFINED_EDGE_SPLIT = True
    m.IS_KNOWLEDGE_GRAPH = False
    m.IS_INDUCTIVE = False
    m.INDUCTIVE_FILTER_MODE = "transductive"
    m.EDGE_SET_MODE = "train_only"
    m.SUPPORTED_VERSIONS = None
    m.PREFERRED_LINK_PRED_EVAL = "sampled_recall"
    m.PREFERRED_EVALUATE_HEAD_PREDICTIONS = False
    m.UNILP_FOLDER = folder
    return m


def build_unilp_registry() -> Dict[str, ModuleType]:
    return {_registry_key(folder): _make_module(folder) for folder in UNILP_ALL_DATASETS}


def register_unilp_datasets(registry: Dict[str, Any]) -> None:
    """Merge UniLP homogeneous LP modules into the global dataset registry."""
    registry.update(build_unilp_registry())
