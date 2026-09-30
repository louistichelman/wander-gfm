"""Resolve G2T TabICL ICL config directories and result labels from env/CLI."""

from __future__ import annotations

import os
from pathlib import Path

from baselines.graphpfn.config_paths import (
    GRAPHPFN_EXP_ROOT,
    GRAPHPFN_ROOT,
    icl_subdir_for_pca_dim,
    resolve_pca_dim,
    run_label_for_pca_dim,
)
from baselines.paths import WANDER_ROOT

G2T_TABICL_ROOT = GRAPHPFN_EXP_ROOT / "g2t" / "tabicl"


def config_root_for_pca_dim(pca_dim: int | None) -> Path:
    return G2T_TABICL_ROOT / icl_subdir_for_pca_dim(pca_dim)


def config_relpath_for_pca_dim(pca_dim: int | None) -> str:
    return str(Path("exp") / "wander" / "g2t" / "tabicl" / icl_subdir_for_pca_dim(pca_dim))


def summary_csv_for_pca_dim(pca_dim: int | None) -> Path:
    return (
        WANDER_ROOT
        / "results"
        / "baselines"
        / "g2t_tabicl"
        / run_label_for_pca_dim(pca_dim)
        / "summary.csv"
    )


def resolve_config_root(
    *,
    pca_dim: str | int | None = None,
    config_root: str | Path | None = None,
) -> Path:
    explicit = config_root or os.environ.get("G2T_CONFIG_ROOT")
    if explicit:
        root = Path(explicit)
        if not root.is_absolute():
            for base in (GRAPHPFN_ROOT, GRAPHPFN_EXP_ROOT, WANDER_ROOT):
                cand = base / root
                if cand.exists():
                    return cand.resolve()
            root = GRAPHPFN_ROOT / root
        return root.resolve()
    return config_root_for_pca_dim(resolve_pca_dim(pca_dim=pca_dim))


def resolve_run_label(
    *,
    pca_dim: str | int | None = None,
    config_root: str | Path | None = None,
) -> str:
    explicit = config_root or os.environ.get("G2T_CONFIG_ROOT")
    if explicit:
        root = Path(explicit)
        name = root.name
        if name == "10":
            return "pca64"
        if name == "10-nopca":
            return "nopca"
        if name.startswith("10-pca"):
            return name.replace("10-", "", 1)
        return name
    return run_label_for_pca_dim(resolve_pca_dim(pca_dim=pca_dim))
