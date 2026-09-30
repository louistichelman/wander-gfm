#!/usr/bin/env python3
"""Generate GraphPFN ICL evaluation.toml configs for Wander datasets.

Writes ``<slug>/evaluation.toml`` (legacy ``split.npz``, 10 ensemble seeds).
Datasets with several Wander NC protocol splits also get
``<slug>/split_{i}/evaluation.toml`` pointing at ``split_{i}.npz``.

Usage (from the repo root):
    python baselines/graphpfn/generate_icl_configs.py
    python baselines/graphpfn/generate_icl_configs.py --no-pca
    python baselines/graphpfn/generate_icl_configs.py --pca-dim 32
    python baselines/graphpfn/generate_icl_configs.py --nofeat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.graphpfn.config_paths import (  # noqa: E402
    config_root_for_ignore_features,
    config_root_for_pca_dim,
)
from baselines.registry import GRAPHPFN_DATASET_SPECS  # noqa: E402
from data.nc_splits import (  # noqa: E402
    default_icl_n_seeds,
    eval_nc_split_indices,
    n_eval_nc_splits,
    split_npz_filename,
    split_run_name,
)

CLASSIC_TEMPLATE = """function = "bin.graphpfn.evaluate.main"
n_seeds = {n_seeds}

[base_config]
amp = true
checkpoint_name = "hf://eremeev-d/graphpfn-1.3/graphpfn-adapters-1_3.pt"
unfreeze_all = true
n_steps = 0
epoch_size = 10
patience = 4
seq_len_pred = 1024
min_train_ratio = 0.75

[base_config.ensemble]
shuffle_features = true
shuffle_targets = {shuffle_targets}
n_members = {n_members}

[base_config.model]
n_random_features = 8

[base_config.model.edge_head]
{d_reduction}
[base_config.data]
cache = false
path = "data/{slug}"
setting = "transductive"
external_split_file = "{split_file}"
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

NOFEAT_CLASSIC_TEMPLATE = """function = "bin.graphpfn.evaluate.main"
n_seeds = {n_seeds}

[base_config]
amp = true
checkpoint_name = "hf://eremeev-d/graphpfn-1.3/graphpfn-adapters-1_3.pt"
unfreeze_all = true
n_steps = 0
epoch_size = 10
patience = 4
seq_len_pred = 1024
min_train_ratio = 0.75

[base_config.ensemble]
shuffle_features = false
shuffle_targets = {shuffle_targets}
n_members = {n_members}

[base_config.model]
n_random_features = 8

[base_config.model.edge_head]

[base_config.data]
cache = false
path = "data/{slug}"
setting = "transductive"
external_split_file = "{split_file}"
{data_score}

[base_config.transform]
labels = false

[base_config.transform.features]
num_policy = "none"

[base_config.optimizer]
type = "AdamW"
lr = 0.0003
weight_decay = 0.0
"""

GRAPHLAND_TEMPLATE = """function = "bin.graphpfn.evaluate.main"
n_seeds = 10

[base_config]
amp = true
checkpoint_name = "hf://eremeev-d/graphpfn-1.3/graphpfn-adapters-1_3.pt"
unfreeze_all = true
n_steps = 0
epoch_size = 10
patience = 4
seq_len_pred = 1024
min_train_ratio = 0.75

[base_config.ensemble]
shuffle_features = true
shuffle_targets = {shuffle_targets}
n_members = {n_members}

[base_config.model]
n_random_features = 8

[base_config.model.edge_head]
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


def iter_icl_eval_targets(spec):
    """Yield ``(subdir, n_seeds, split_file)`` for ICL evaluation.toml files.

    Always writes the legacy ``<slug>/evaluation.toml`` (``split.npz``, 10
    ensemble seeds). Multi-split datasets also get ``<slug>/split_{i}/`` with
    ``n_seeds=1`` so the reporting pass averages Wander protocol splits.
    """
    n = n_eval_nc_splits(spec.registry_key)
    yield spec.slug, 10, "split.npz"
    if n <= 1:
        return
    for split_index in eval_nc_split_indices(spec.registry_key):
        subdir = split_run_name(spec.registry_key, split_index, stem=spec.slug)
        split_file = split_npz_filename(spec.registry_key, split_index)
        yield subdir, default_icl_n_seeds(spec.registry_key), split_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GraphPFN ICL evaluation.toml files.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--no-pca",
        action="store_true",
        help="Omit d_reduction (full feature dimension). Writes to icl/10-nopca/.",
    )
    group.add_argument(
        "--pca-dim",
        type=int,
        default=64,
        help="PCA target dimension (default: 64 → icl/10/). Use with --no-pca disabled.",
    )
    parser.add_argument(
        "--nofeat",
        action="store_true",
        help="Generate topology-only configs (ignore node features) under icl/10-nofeat/.",
    )
    args = parser.parse_args()

    if args.nofeat:
        config_root = config_root_for_ignore_features()
        specs = [s for s in GRAPHPFN_DATASET_SPECS if s.ignore_features]
        if not specs:
            raise SystemExit("No ignore_features dataset specs found.")
        d_reduction = "\n"
    else:
        pca_dim = None if args.no_pca else args.pca_dim
        config_root = config_root_for_pca_dim(pca_dim)
        specs = [s for s in GRAPHPFN_DATASET_SPECS if not s.ignore_features]
        d_reduction = _d_reduction_block(pca_dim)

    for spec in specs:
        shuffle_targets = "false" if spec.is_binary else "true"
        data_score = 'score = "roc-auc"' if spec.is_binary else ""
        graphland = (not args.nofeat) and spec.split_kind == "graphland_rl"
        for subdir, n_seeds, split_file in iter_icl_eval_targets(spec):
            if graphland and subdir != spec.slug:
                continue
            out_dir = config_root / subdir
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "evaluation.toml"
            if args.nofeat:
                content = NOFEAT_CLASSIC_TEMPLATE.format(
                    slug=spec.slug,
                    shuffle_targets=shuffle_targets,
                    n_members=spec.n_members,
                    n_seeds=n_seeds,
                    split_file=split_file,
                    data_score=data_score,
                )
            elif graphland:
                content = GRAPHLAND_TEMPLATE.format(
                    slug=spec.slug,
                    shuffle_targets=shuffle_targets,
                    n_members=spec.n_members,
                    d_reduction=d_reduction,
                    data_score=data_score,
                )
            else:
                content = CLASSIC_TEMPLATE.format(
                    slug=spec.slug,
                    shuffle_targets=shuffle_targets,
                    n_members=spec.n_members,
                    n_seeds=n_seeds,
                    split_file=split_file,
                    d_reduction=d_reduction,
                    data_score=data_score,
                )
            out_path.write_text(content)
            print(f"Wrote {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
