"""Register AnyGraph link-prediction datasets into DATASET_REGISTRY.

Paper Table 3 homogeneous LP graphs. Registry keys are ANYGRAPH_<NAME>
(uppercased, non-alphanumerics -> _) to avoid clashing with the existing
node-classification entries (CORA, PUBMED, ...).

Bipartite (user x item) layout is determined at load time from a rectangular
trn_mat.pkl (see loader.load_anygraph_bundle); there is no hardcoded
bipartite flag on registry modules.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, Dict, Optional

from .loader import load_anygraph_bundle

# AnyGraph warm-val LP datasets: every val edge is already in train, so Recall@K on
# val is always 0 after train-positive filtering. Monitor test during training val.
_WARM_VAL_USE_TEST_EVAL = {
    "products_home", "ddi",
}

# Paper Table 3 / 6 homogeneous LP graphs.
_DATASETS = [
    "products_home",
    "pubmed",
    "citeseer",
    "p2p-Gnutella06",
    "soc-Epinions1",
    "email-Enron",
    "cora",
    "CS",
    "proteins_spec1",
    "ddi",
]


def _registry_key(folder: str) -> str:
    return "ANYGRAPH_" + folder.upper().replace("-", "_").replace(".", "_")


def _make_module(folder: str) -> ModuleType:
    key = _registry_key(folder)

    def loader(
        data_dir: str,
        seed: int = 42,
        add_inverse_edges_kgs: bool = True,
        dataset_version: Optional[str] = None,
    ):
        del seed, dataset_version
        return load_anygraph_bundle(
            data_dir,
            folder,
            display_name=key,
            add_inverse_edges_kgs=add_inverse_edges_kgs,
        )

    m = ModuleType(key)
    m.NAME = key
    m.load_graph_bundle = loader
    m.load_base_dataset = None  # type: ignore[assignment]
    m.HAS_NODE_FEATURES = True
    m.HAS_NODE_LABELS = False
    m.HAS_PREDEFINED_NODE_SPLIT = False
    m.HAS_PREDEFINED_EDGE_SPLIT = True
    # Rectangular adjacency -> directed user|item bundle at load time (see loader).
    m.IS_KNOWLEDGE_GRAPH = False
    m.IS_INDUCTIVE = False
    m.INDUCTIVE_FILTER_MODE = "transductive"
    m.EDGE_SET_MODE = "train_only"
    m.SUPPORTED_VERSIONS = None
    # AnyGraph datasets are compared with the Recall@K (tail-only) protocol.
    m.PREFERRED_LINK_PRED_EVAL = "recall"
    m.PREFERRED_EVALUATE_HEAD_PREDICTIONS = False
    if folder in _WARM_VAL_USE_TEST_EVAL:
        m.PREFERRED_EVAL_SPLIT = "test"
    return m


def build_anygraph_registry() -> Dict[str, ModuleType]:
    return {_registry_key(folder): _make_module(folder) for folder in _DATASETS}


def register_anygraph_datasets(registry: Dict[str, Any]) -> None:
    """Merge AnyGraph link-prediction modules into the global DATASET_REGISTRY."""
    registry.update(build_anygraph_registry())
