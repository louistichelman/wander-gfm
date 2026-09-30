#!/usr/bin/env python3
"""Generate G2T+TabICL ICL evaluation.toml configs for Wander datasets.

Mirrors the GraphPFN ICL sweep (same splits, PCA64 / no-PCA), using the
simplified node-feature-only G2T template aligned with Wander evals.

Usage (from the repo root):
    python baselines/graphpfn/generate_g2t_tabicl_configs.py
    python baselines/graphpfn/generate_g2t_tabicl_configs.py --no-pca
    python baselines/graphpfn/generate_g2t_tabicl_configs.py --pca-dim 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.graphpfn.g2t_config_paths import config_root_for_pca_dim  # noqa: E402
from baselines.registry import GRAPHPFN_DATASET_SPECS  # noqa: E402

CLASSIC_TEMPLATE = """function = "bin.g2t.g2t_fm.main"
n_seeds = 10

[base_config]
patience = 16
amp = true
n_epochs = 0
epoch_size = 10
min_train_ratio = 0.75
seq_len_pred = 1024
finetune_mode = "full"

[base_config.tfm]
name = "tabicl"

[base_config.ensemble]
shuffle_features = true
shuffle_targets = {shuffle_targets}
n_members = {n_members}
{d_reduction}
[base_config.data]
cache = false
path = "data/{slug}"
setting = "transductive"
external_split_file = "split.npz"
{data_score}

[base_config.transform]
labels = false

[base_config.transform.features]
num_policy = "standard"

[base_config.optimizer]
type = "AdamW"
lr = 0.0003
weight_decay = 0.0
"""

GRAPHLAND_TEMPLATE = """function = "bin.g2t.g2t_fm.main"
n_seeds = 10

[base_config]
patience = 16
amp = true
n_epochs = 0
epoch_size = 10
min_train_ratio = 0.75
seq_len_pred = 1024
finetune_mode = "full"

[base_config.tfm]
name = "tabicl"

[base_config.ensemble]
shuffle_features = true
shuffle_targets = {shuffle_targets}
n_members = {n_members}
{d_reduction}
[base_config.data]
cache = false
path = "data/{slug}"
setting = "transductive"
internal_split_name = "RL"
{data_score}

[base_config.transform]
labels = false

[base_config.transform.features]
seed = 0
cat_policy = "ordinal"
num_policy = "quantile-normal"
frac_policy = "quantile-uniform"

[base_config.optimizer]
type = "AdamW"
lr = 0.0003
weight_decay = 0.0
"""


def _d_reduction_block(pca_dim: int | None) -> str:
    if pca_dim is None:
        return "\n"
    return f"""
[base_config.d_reduction]
method = "pca"
d = {pca_dim}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate G2T TabICL ICL evaluation.toml files.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--no-pca",
        action="store_true",
        help="Omit d_reduction (full feature dimension). Writes to g2t/tabicl/icl/10-nopca/.",
    )
    group.add_argument(
        "--pca-dim",
        type=int,
        default=64,
        help="PCA target dimension (default: 64 → g2t/tabicl/icl/10/).",
    )
    args = parser.parse_args()

    pca_dim = None if args.no_pca else args.pca_dim
    config_root = config_root_for_pca_dim(pca_dim)
    d_reduction = _d_reduction_block(pca_dim)
    specs = [s for s in GRAPHPFN_DATASET_SPECS if not s.ignore_features]

    for spec in specs:
        out_dir = config_root / spec.slug
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "evaluation.toml"
        shuffle_targets = "false" if spec.is_binary else "true"
        data_score = 'score = "roc-auc"' if spec.is_binary else ""
        template = GRAPHLAND_TEMPLATE if spec.split_kind == "graphland_rl" else CLASSIC_TEMPLATE
        content = template.format(
            slug=spec.slug,
            shuffle_targets=shuffle_targets,
            n_members=spec.n_members,
            d_reduction=d_reduction,
            data_score=data_score,
        )
        out_path.write_text(content)
        print(f"Wrote {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
