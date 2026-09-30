#!/usr/bin/env python3
"""Generate GraphPFN finetune tuning.toml configs for Wander datasets.

Uses the official GraphPFN finetune recipe: classic preprocessing, PCA-64,
standard 10-value LR grid, full-model finetuning with early stopping.

Usage (from the repo root):
    python baselines/graphpfn/generate_finetune_configs.py
    python baselines/graphpfn/generate_finetune_configs.py --datasets cora,citeseer
    python baselines/graphpfn/generate_finetune_configs.py --pca-dim 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.graphpfn.config_paths import config_root_for_finetune  # noqa: E402
from baselines.registry import (  # noqa: E402
    GRAPHPFN_DATASET_SPECS,
    GraphPFNDatasetSpec,
    graphpfn_multiclass_nc_slugs,
)

STANDARD_LR_GRID = """
        4.999998054699972e-06,
        8.340500244230498e-06,
        1.3912793292547576e-05,
        2.320794010302052e-05,
        3.871318040182814e-05,
        6.457747804233804e-05,
        0.00010772173118311912,
        0.00017969068721868098,
        0.00029974215431138873,
        0.0005000000819563866,"""

CLASSIC_FINETUNE_TUNING_TEMPLATE = """seed = 0
n_trials = 100
sampler_type = "BruteForceSampler"
function = "bin.graphpfn.evaluate.main"

[space]
seed = 0
amp = true
checkpoint_name = "hf://eremeev-d/graphpfn-1.3/graphpfn-adapters-1_3.pt"
unfreeze_all = true
n_steps = -1
epoch_size = 10
patience = 4
seq_len_pred = 1024
min_train_ratio = 0.75

[space.ensemble]
shuffle_features = true
shuffle_targets = {shuffle_targets}
n_members = {n_members}

[space.model]
n_random_features = 8

[space.model.edge_head]
{d_reduction}
[space.data]
cache = false
path = "data/{slug}"
setting = "transductive"
external_split_file = "split.npz"
{data_score}

[space.transform]
labels = false

[space.transform.features]
num_policy = "standard"

[space.optimizer]
type = "AdamW"
lr = [
    "_tune_",
    "categorical",
    [
{lr_grid}
    ],
]
weight_decay = 0.0
"""

NOFEAT_FINETUNE_TUNING_TEMPLATE = """seed = 0
n_trials = 100
sampler_type = "BruteForceSampler"
function = "bin.graphpfn.evaluate.main"

[space]
seed = 0
amp = true
checkpoint_name = "hf://eremeev-d/graphpfn-1.3/graphpfn-adapters-1_3.pt"
unfreeze_all = true
n_steps = -1
epoch_size = 10
patience = 4
seq_len_pred = 1024
min_train_ratio = 0.75

[space.ensemble]
shuffle_features = false
shuffle_targets = {shuffle_targets}
n_members = {n_members}

[space.model]
n_random_features = 8

[space.model.edge_head]

[space.data]
cache = false
path = "data/{slug}"
setting = "transductive"
external_split_file = "split.npz"
{data_score}

[space.transform]
labels = false

[space.transform.features]
num_policy = "none"

[space.optimizer]
type = "AdamW"
lr = [
    "_tune_",
    "categorical",
    [
{lr_grid}
    ],
]
weight_decay = 0.0
"""

GRAPHLAND_FINETUNE_TUNING_TEMPLATE = """seed = 0
n_trials = 100
sampler_type = "BruteForceSampler"
function = "bin.graphpfn.evaluate.main"

[space]
seed = 0
amp = true
checkpoint_name = "hf://eremeev-d/graphpfn-1.3/graphpfn-adapters-1_3.pt"
unfreeze_all = true
n_steps = -1
epoch_size = 10
patience = 4
seq_len_pred = 1024
min_train_ratio = 0.75

[space.ensemble]
shuffle_features = true
shuffle_targets = {shuffle_targets}
n_members = {n_members}

[space.model]
n_random_features = 8

[space.model.edge_head]
{d_reduction}
[space.data]
cache = false
path = "data/{slug}"
setting = "transductive"
internal_split_name = "RL"
{data_score}

[space.transform]
labels = false

[space.transform.features]
seed = 0
cat_policy = "ordinal"
num_policy = "quantile-normal"
frac_policy = "quantile-uniform"

[space.optimizer]
type = "AdamW"
lr = [
    "_tune_",
    "categorical",
    [
{lr_grid}
    ],
]
weight_decay = 0.0
"""


def _d_reduction_block(pca_dim: int | None) -> str:
    if pca_dim is None:
        return "\n"
    return f"""
[space.d_reduction]
method = "pca"
d = {pca_dim}
"""


def _select_specs(slugs: list[str]) -> list[GraphPFNDatasetSpec]:
    slug_set = {s.lower() for s in slugs}
    specs = [spec for spec in GRAPHPFN_DATASET_SPECS if spec.slug in slug_set]
    found = {spec.slug for spec in specs}
    missing = slug_set - found
    if missing:
        raise SystemExit(f"Unknown dataset slug(s): {', '.join(sorted(missing))}")
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate GraphPFN finetune tuning.toml files."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help=(
            "Comma-separated dataset slugs. Default is classic multiplex unless "
            "--multiclass-nc is set."
        ),
    )
    parser.add_argument(
        "--multiclass-nc",
        action="store_true",
        help=(
            "Generate featured multiclass NC configs, excluding multiplex / MUX "
            "families (PCA-64 by default)."
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--no-pca",
        action="store_true",
        help="Omit d_reduction (full feature dimension).",
    )
    group.add_argument(
        "--pca-dim",
        type=int,
        default=64,
        help="PCA target dimension (default: 64).",
    )
    parser.add_argument(
        "--nofeat",
        action="store_true",
        help="Generate topology-only finetune configs under finetune/10-nofeat/.",
    )
    args = parser.parse_args()

    if args.datasets.strip():
        requested = [s.strip() for s in args.datasets.split(",") if s.strip()]
    elif args.multiclass_nc:
        requested = graphpfn_multiclass_nc_slugs(featured_only=True)
    elif args.nofeat:
        requested = [s.slug for s in GRAPHPFN_DATASET_SPECS if s.ignore_features]
    else:
        requested = graphpfn_multiclass_nc_slugs(featured_only=True)

    if args.nofeat:
        config_root = config_root_for_finetune(ignore_features=True)
        specs = _select_specs(requested)
        d_reduction = ""
    else:
        pca_dim = None if args.no_pca else args.pca_dim
        config_root = config_root_for_finetune(pca_dim)
        specs = _select_specs(requested)
        d_reduction = _d_reduction_block(pca_dim)

    for spec in specs:
        if args.nofeat and not spec.ignore_features:
            raise SystemExit(
                f"{spec.slug}: --nofeat requires an ignore_features dataset slug "
                f"(e.g. {spec.slug}-nofeat)."
            )
        if args.nofeat:
            template = NOFEAT_FINETUNE_TUNING_TEMPLATE
        elif spec.split_kind == "graphland_rl":
            template = GRAPHLAND_FINETUNE_TUNING_TEMPLATE
        else:
            template = CLASSIC_FINETUNE_TUNING_TEMPLATE
        out_dir = config_root / spec.slug
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "tuning.toml"
        shuffle_targets = "false" if spec.is_binary else "true"
        data_score = 'score = "roc-auc"' if spec.is_binary else ""
        content = template.format(
            slug=spec.slug,
            shuffle_targets=shuffle_targets,
            n_members=spec.n_members,
            d_reduction=d_reduction,
            data_score=data_score,
            lr_grid=STANDARD_LR_GRID,
        )
        out_path.write_text(content)
        print(f"Wrote {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
