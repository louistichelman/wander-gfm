"""Resolve GraphPFN ICL config directories and result labels from env/CLI."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_WANDER_ROOT = Path(__file__).resolve().parents[2]
if str(_WANDER_ROOT) not in sys.path:
    sys.path.insert(0, str(_WANDER_ROOT))

from baselines.paths import WANDER_ROOT, GRAPHPFN_CONFIG_ROOT, GRAPHPFN_ROOT
from data.nc_splits import default_icl_n_seeds, split_run_name

GRAPHPFN_EXP_ROOT = GRAPHPFN_CONFIG_ROOT
_GRAPHPFN_REL_PREFIX = Path("exp") / "wander"


def _normalize_pca_dim(raw: str | None) -> int | None:
    if raw is None:
        return 64
    value = raw.strip().lower()
    if value in {"", "64", "default"}:
        return 64
    if value in {"0", "none", "off", "no", "false", "nopca"}:
        return None
    return int(value)


def icl_subdir_for_pca_dim(pca_dim: int | None) -> str:
    if pca_dim is None:
        return "icl/10-nopca"
    if pca_dim == 64:
        return "icl/10"
    return f"icl/10-pca{pca_dim}"


NOFEAT_ICL_SUBDIR = "icl/10-nofeat"
FINETUNE_SUBDIR = "finetune/10"
NOFEAT_FINETUNE_SUBDIR = "finetune/10-nofeat"


def finetune_subdir_for_pca_dim(pca_dim: int | None) -> str:
    if pca_dim is None:
        return "finetune/10-nopca"
    if pca_dim == 64:
        return FINETUNE_SUBDIR
    return f"finetune/10-pca{pca_dim}"


def config_root_for_finetune(
    pca_dim: int | None = 64,
    *,
    ignore_features: bool = False,
) -> Path:
    if ignore_features:
        return GRAPHPFN_EXP_ROOT / NOFEAT_FINETUNE_SUBDIR
    return GRAPHPFN_EXP_ROOT / finetune_subdir_for_pca_dim(pca_dim)


def config_root_for_ignore_features() -> Path:
    return GRAPHPFN_EXP_ROOT / NOFEAT_ICL_SUBDIR


def run_label_for_ignore_features() -> str:
    return "nofeat"


def run_label_for_pca_dim(pca_dim: int | None) -> str:
    if pca_dim is None:
        return "nopca"
    return f"pca{pca_dim}"


def config_root_for_pca_dim(pca_dim: int | None) -> Path:
    return GRAPHPFN_EXP_ROOT / icl_subdir_for_pca_dim(pca_dim)


def config_relpath_for_pca_dim(pca_dim: int | None) -> str:
    """Path relative to the GraphPFN checkout (via ``exp/wander`` symlink)."""
    return str(_GRAPHPFN_REL_PREFIX / icl_subdir_for_pca_dim(pca_dim))


def config_relpath_for_ignore_features() -> str:
    return str(_GRAPHPFN_REL_PREFIX / NOFEAT_ICL_SUBDIR)


def icl_config_tree_relpath(pca_dim: int | None, *, ignore_features: bool = False) -> str:
    if ignore_features:
        return config_relpath_for_ignore_features()
    return config_relpath_for_pca_dim(pca_dim)


def icl_eval_relpath(
    spec,
    pca_dim: int | None,
    split_index: int = 0,
    *,
    ignore_features: bool | None = None,
) -> str:
    """``exp/wander/icl/.../<slug>[/split_i]/evaluation.toml``."""
    nofeat = spec.ignore_features if ignore_features is None else ignore_features
    tree = icl_config_tree_relpath(pca_dim, ignore_features=nofeat)
    sub = split_run_name(spec.registry_key, split_index, stem=spec.slug)
    return f"{tree}/{sub}/evaluation.toml"


def icl_n_seeds_for_spec(spec, n_seeds: int | None = None) -> int:
    if n_seeds is not None:
        return int(n_seeds)
    return default_icl_n_seeds(spec.registry_key)


def config_relpath_for_finetune(
    pca_dim: int | None,
    *,
    ignore_features: bool = False,
) -> str:
    """Path relative to the GraphPFN checkout for finetune ``tuning.toml`` files."""
    if ignore_features:
        return str(_GRAPHPFN_REL_PREFIX / NOFEAT_FINETUNE_SUBDIR)
    return str(_GRAPHPFN_REL_PREFIX / finetune_subdir_for_pca_dim(pca_dim))


def summary_csv_for_pca_dim(pca_dim: int | None) -> Path:
    return (
        WANDER_ROOT
        / "results"
        / "baselines"
        / "graphpfn"
        / run_label_for_pca_dim(pca_dim)
        / "summary.csv"
    )


def splits_csv_for_pca_dim(pca_dim: int | None) -> Path:
    return summary_csv_for_pca_dim(pca_dim).with_name("splits.csv")


def _coerce_config_root(raw: str | Path) -> Path:
    root = Path(raw)
    if root.is_absolute():
        return root.resolve()
    for base in (GRAPHPFN_ROOT, GRAPHPFN_CONFIG_ROOT, WANDER_ROOT):
        cand = base / root
        if cand.exists():
            return cand.resolve()
    return (GRAPHPFN_ROOT / root).resolve()


def resolve_pca_dim(
    *,
    pca_dim: str | int | None = None,
    config_root: str | Path | None = None,
) -> int | None:
    if config_root is not None:
        root = _coerce_config_root(config_root)
        name = root.name
        if name == "10-nopca":
            return None
        if name == "10":
            return 64
        if name.startswith("10-pca") and name[8:].isdigit():
            return int(name[8:])
        raise ValueError(f"Cannot infer PCA setting from config root: {root}")
    if pca_dim is None:
        return _normalize_pca_dim(os.environ.get("GRAPHPFN_PCA_DIM"))
    if isinstance(pca_dim, str):
        return _normalize_pca_dim(pca_dim)
    if pca_dim == 0:
        return None
    return pca_dim


def resolve_config_root(
    *,
    pca_dim: str | int | None = None,
    config_root: str | Path | None = None,
    ignore_features: bool = False,
) -> Path:
    explicit = config_root or os.environ.get("GRAPHPFN_CONFIG_ROOT")
    if explicit:
        return _coerce_config_root(explicit)
    if ignore_features or os.environ.get("GRAPHPFN_IGNORE_FEATURES", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        return config_root_for_ignore_features()
    return config_root_for_pca_dim(resolve_pca_dim(pca_dim=pca_dim))


def resolve_run_label(
    *,
    pca_dim: str | int | None = None,
    config_root: str | Path | None = None,
    ignore_features: bool = False,
) -> str:
    explicit = config_root or os.environ.get("GRAPHPFN_CONFIG_ROOT")
    if explicit:
        root = Path(explicit)
        name = root.name
        if name == "10":
            return "pca64"
        if name == "10-nopca":
            return "nopca"
        if name == "10-nofeat":
            return "nofeat"
        if name == "10-train0.4":
            return "train0.4"
        if name == "10-train0.1":
            return "train0.1"
        if name.startswith("10-pca"):
            return name.replace("10-", "", 1)
        return name
    if ignore_features or os.environ.get("GRAPHPFN_IGNORE_FEATURES", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        return run_label_for_ignore_features()
    return run_label_for_pca_dim(resolve_pca_dim(pca_dim=pca_dim))
