"""Light hyperparameter grid for AnyGraph LP baselines (BUDDY, NBFNet).

Paper Hom-LP grids (Table 3 datasets):
  BUDDY  — 4 architecture × 3 negative/weight settings = 12
  NBFNet — 6 architecture × 3 negative/weight settings = 18, max 40 epochs
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Table 3 homogeneous reporting set.
HOMOGENEOUS_DATASETS: Tuple[str, ...] = (
    "ANYGRAPH_CITESEER",
    "ANYGRAPH_CORA",
    "ANYGRAPH_PUBMED",
    "ANYGRAPH_CS",
    "ANYGRAPH_DDI",
    "ANYGRAPH_P2P_GNUTELLA06",
    "ANYGRAPH_EMAIL_ENRON",
    "ANYGRAPH_PROTEINS_SPEC1",
    "ANYGRAPH_SOC_EPINIONS1",
    "ANYGRAPH_PRODUCTS_HOME",
)
GRID_DATASETS: Tuple[str, ...] = HOMOGENEOUS_DATASETS
REPORTING_DATASETS: Tuple[str, ...] = HOMOGENEOUS_DATASETS

FEATURED_VAL = (
    "ANYGRAPH_CITESEER",
    "ANYGRAPH_CORA",
    "ANYGRAPH_PUBMED",
    "ANYGRAPH_CS",
)
NOFEAT_VAL = (
    "ANYGRAPH_P2P_GNUTELLA06",
    "ANYGRAPH_EMAIL_ENRON",
    "ANYGRAPH_PROTEINS_SPEC1",
    "ANYGRAPH_SOC_EPINIONS1",
)
WARM_VAL = ("ANYGRAPH_DDI", "ANYGRAPH_PRODUCTS_HOME")
FEATURED_WARM = ("ANYGRAPH_PRODUCTS_HOME",)
NOFEAT_WARM = ("ANYGRAPH_DDI",)

LARGE_DATASETS = {
    "ANYGRAPH_DDI",
    "ANYGRAPH_EMAIL_ENRON",
    "ANYGRAPH_PROTEINS_SPEC1",
    "ANYGRAPH_SOC_EPINIONS1",
    "ANYGRAPH_PRODUCTS_HOME",
    "ANYGRAPH_PUBMED",
}

# Shared training knobs (not swept).
SHARED = {
    "num_negative": 32,
    "seed": 0,
    "max_monitor_queries": 512,
    "pca_target_dim": 64,
}
NBFNET_MAX_EPOCHS = 40

BUDDY_ARCH_CONFIGS: List[Dict[str, Any]] = [
    {"id": "h128_lr1e-3", "hidden_channels": 128, "lr": 1e-3},
    {"id": "h128_lr5e-3", "hidden_channels": 128, "lr": 5e-3},
    {"id": "h256_lr1e-3", "hidden_channels": 256, "lr": 1e-3},
    {"id": "h256_lr5e-3", "hidden_channels": 256, "lr": 5e-3},
]

NBFNET_ARCH_CONFIGS: List[Dict[str, Any]] = [
    {"id": "d3_lr1e-3", "hidden_dims": (32, 32, 32), "lr": 1e-3},
    {"id": "d3_lr5e-3", "hidden_dims": (32, 32, 32), "lr": 5e-3},
    {"id": "d5_lr1e-3", "hidden_dims": (32, 32, 32, 32, 32), "lr": 1e-3},
    {"id": "d5_lr5e-3", "hidden_dims": (32, 32, 32, 32, 32), "lr": 5e-3},
    {"id": "d6_lr1e-3", "hidden_dims": (32, 32, 32, 32, 32, 32), "lr": 1e-3},
    {"id": "d6_lr5e-3", "hidden_dims": (32, 32, 32, 32, 32, 32), "lr": 5e-3},
]

# 1-neg is already 1:1 under mean BCE; 32/256 use Wander-style equal pos/neg-set weight.
NEG_VARIANTS: List[Dict[str, Any]] = [
    {"id": "neg1", "num_negative": 1, "equal_pos_neg_weight": False},
    {"id": "neg32_eq", "num_negative": 32, "equal_pos_neg_weight": True},
    {"id": "neg256_eq", "num_negative": 256, "equal_pos_neg_weight": True},
]
NBFNET_NEG_VARIANTS = NEG_VARIANTS


def _neg_grid_configs(arch_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for arch in arch_configs:
        for var in NEG_VARIANTS:
            row = dict(arch)
            row["id"] = f"{arch['id']}_{var['id']}"
            row["num_negative"] = int(var["num_negative"])
            row["equal_pos_neg_weight"] = bool(var["equal_pos_neg_weight"])
            out.append(row)
    return out


BUDDY_CONFIGS: List[Dict[str, Any]] = _neg_grid_configs(BUDDY_ARCH_CONFIGS)
NBFNET_CONFIGS: List[Dict[str, Any]] = _neg_grid_configs(NBFNET_ARCH_CONFIGS)

METHOD_CONFIGS = {
    "buddy": BUDDY_CONFIGS,
    "nbfnet": NBFNET_CONFIGS,
}


def configs_for(method: str) -> List[Dict[str, Any]]:
    key = method.lower()
    if key not in METHOD_CONFIGS:
        raise KeyError(f"unknown method {method!r}; expected buddy|nbfnet")
    return METHOD_CONFIGS[key]


def normalize_dataset_keys(names: List[str]) -> List[str]:
    """Uppercase registry keys; keep ANYGRAPH_ prefix as-is."""
    out: List[str] = []
    seen = set()
    for name in names:
        lower = name.lower()
        if name.startswith("ANYGRAPH_"):
            key = name
        elif lower.startswith("anygraph_"):
            key = "ANYGRAPH_" + name[len("anygraph_") :].upper().replace("-", "_")
        else:
            key = name.upper()
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def parse_dataset_list(raw: str) -> List[str]:
    names = [x.strip() for x in raw.replace(" ", ",").split(",") if x.strip()]
    return normalize_dataset_keys(names)


def nbfnet_neg_grid_configs() -> List[Dict[str, Any]]:
    """Paper NBFNet grid: architecture × negative/weight (18 settings)."""
    return list(NBFNET_CONFIGS)


def buddy_neg_grid_configs() -> List[Dict[str, Any]]:
    """Paper BUDDY grid: architecture × negative/weight (12 settings)."""
    return list(BUDDY_CONFIGS)


def val_group_for(dataset: str) -> Tuple[str, ...]:
    if dataset in FEATURED_WARM or dataset in FEATURED_VAL:
        return FEATURED_VAL
    return NOFEAT_VAL


def is_warm_val(dataset: str) -> bool:
    return dataset in WARM_VAL


def batches_per_epoch_for(dataset: str, method: str) -> Optional[int]:
    """Cap large-graph epochs so the paper grid stays cheap."""
    if dataset in LARGE_DATASETS:
        return 256
    if method == "nbfnet" and dataset == "ANYGRAPH_CS":
        return 256
    return None
