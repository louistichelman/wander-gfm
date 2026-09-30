"""FB15k-237 dataset definition (transductive KG bundle)."""

from typing import Optional

from ..graph_bundle import GraphBundle
from .pretrain_kg_loaders import load_fb15k237_bundle

# Dataset metadata
NAME = "FB15k-237"
HAS_NODE_FEATURES = False
HAS_NODE_LABELS = False
HAS_PREDEFINED_NODE_SPLIT = False
HAS_PREDEFINED_EDGE_SPLIT = True
IS_KNOWLEDGE_GRAPH = True
IS_INDUCTIVE = False
INDUCTIVE_FILTER_MODE = "transductive"
EDGE_SET_MODE = "all_positives"
SUPPORTED_VERSIONS = None
PREFERRED_LINK_PRED_EVAL = "mrr"
PREFERRED_EVALUATE_HEAD_PREDICTIONS = True


def load_graph_bundle(
    data_dir: str,
    seed: int = 42,
    add_inverse_edges_kgs: bool = True,
    dataset_version: Optional[str] = None,
) -> GraphBundle:
    del seed, dataset_version
    return load_fb15k237_bundle(
        data_dir,
        add_inverse_edges_kgs=add_inverse_edges_kgs,
        edge_set_mode=EDGE_SET_MODE,
    )


load_base_dataset = None  # type: ignore[assignment]
