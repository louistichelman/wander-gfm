"""CoDEx-Medium dataset definition (transductive KG bundle).

CoDEx-Medium is a knowledge graph completion benchmark extracted from
Wikidata/Wikipedia (Safavi & Koutra, EMNLP 2020).
17,050 entities, 51 relations, ~206k triples.

Data is downloaded from the official CoDEx GitHub repository:
https://github.com/tsafavi/codex
"""

from typing import Optional

from ..graph_bundle import GraphBundle
from .pretrain_kg_loaders import load_codex_medium_bundle

# Dataset metadata
NAME = "CoDEx-Medium"
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
    return load_codex_medium_bundle(
        data_dir,
        add_inverse_edges_kgs=add_inverse_edges_kgs,
        edge_set_mode=EDGE_SET_MODE,
    )


load_base_dataset = None  # type: ignore[assignment]
