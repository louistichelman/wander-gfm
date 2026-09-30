#!/usr/bin/env python3
"""Generate and cache synthetic NC eval graphs (CYCLES, GRIDS).

Usage (from the repo root):
    python scripts/data/generate_motif_datasets.py
    python scripts/data/generate_motif_datasets.py --datasets GRIDS
    python scripts/data/generate_motif_datasets.py --force
    python scripts/data/generate_motif_datasets.py --data_dir /path/to/raw_data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from data.datasets import cycles  # noqa: E402
from data.datasets import grids  # noqa: E402
from data.datasets import motif_er  # noqa: E402


def _print_stats(meta: dict) -> None:
    key = meta["registry_key"]
    print(f"=== {key} ===")
    print(f"  cache:          {meta['cache']}")
    print(f"  n:              {int(meta['n'])}")
    if "er_p" in meta:
        print(f"  er_p:           {meta['er_p']}")
    if "degree" in meta:
        print(f"  degree:         {meta['degree']}")
    if "motif_k" in meta:
        print(f"  motif_k:        {int(meta['motif_k'])}")
    if "n_motifs_c3" in meta:
        print(f"  n_motifs:       C3={int(meta['n_motifs_c3'])} C4={int(meta['n_motifs_c4'])} C5={int(meta['n_motifs_c5'])}")
    elif "n_motifs" in meta:
        print(f"  n_motifs:       {int(meta['n_motifs'])}")
    elif "n_triangles" in meta:
        print(f"  n_triangles:    {int(meta['n_triangles'])}")
    if "grid_h" in meta:
        print(
            f"  grids:          {int(meta['n_grids'])} x {int(meta['grid_h'])}x{int(meta['grid_w'])} "
            f"(diameter={int(meta['diameter'])})"
        )
        if "n_grids_with_train" in meta:
            extra = ""
            if "train_per_grid" in meta:
                extra = f" train_per_grid={int(meta['train_per_grid'])}"
            print(f"  grids w/ train: {int(meta['n_grids_with_train'])}{extra}")
        if "max_hops_to_train" in meta:
            print(
                f"  hops to train:  mean={meta['mean_hops_to_train']:.2f} "
                f"max={int(meta['max_hops_to_train'])}"
            )
    if meta.get("has_node_features") is False:
        print("  features:       none")
    if "mean_degree" in meta:
        if "mean_degree_pos" in meta:
            print(
                f"  mean degree:    all={meta['mean_degree']:.4f} "
                f"pos={meta['mean_degree_pos']:.4f} neg={meta['mean_degree_neg']:.4f}"
            )
        else:
            print(f"  mean degree:    {meta['mean_degree']:.4f}")
    if "class_counts" in meta:
        cc = meta["class_counts"]
        print(
            "  class counts:   "
            + " ".join(f"{k}={int(v)}" for k, v in cc.items())
        )
    if "train_per_class" in meta:
        tpc = meta["train_per_class"]
        print(
            "  train/class:    "
            + " ".join(f"{k}={int(v)}" for k, v in tpc.items())
        )
    print(f"  n_edges (undir):{int(meta['n_edges'])}")
    if "n_pos" in meta:
        print(
            f"  labels global:  pos={int(meta['n_pos'])} neg={int(meta['n_neg'])} "
            f"pos_rate={meta['pos_rate']:.4f}"
        )
        print(
            f"  labels train:   pos={int(meta['n_pos_train'])} neg={int(meta['n_neg_train'])} "
            f"pos_rate={meta['pos_rate_train']:.4f} "
            f"(n_train={int(meta['n_train'])})"
        )
    print(f"  split sizes:    train={int(meta['n_train'])} val={meta['n_val']} test={meta['n_test']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(REPO_ROOT / "raw_data"),
        help="Root data directory (default: Wander/raw_data)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild caches even if data.pt already exists",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="CYCLES,GRIDS",
        help="Comma-separated registry keys (CYCLES, GRIDS)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    modules = {
        "CYCLES": cycles,
        "GRIDS": grids,
    }
    keys = [k.strip().upper() for k in args.datasets.split(",") if k.strip()]
    for key in keys:
        if key not in modules:
            print(f"ERROR: unknown dataset {key!r}; choose from {sorted(modules)}", file=sys.stderr)
            return 1
        mod = modules[key]
        print(f"Generating {key} (force={args.force}) ...")
        data = mod.load_base_dataset(str(data_dir), force=args.force)
        cache = motif_er.cache_path(str(data_dir), key)
        if not cache.is_file():
            print(f"ERROR: cache missing after build: {cache}", file=sys.stderr)
            return 1
        if hasattr(mod, "dataset_meta"):
            meta = mod.dataset_meta(str(data_dir))
            meta["cache"] = str(cache)
        else:
            stats = motif_er.label_stats(data.y, data.train_mask)
            meta = {
                "registry_key": key,
                "n": mod.N_NODES,
                "er_p": getattr(mod, "ER_P", None),
                "n_edges": motif_er.num_undirected_edges(data.edge_index),
                "cache": str(cache),
                **stats,
                "n_val": int(data.val_mask.sum().item()),
                "n_test": int(data.test_mask.sum().item()),
            }
        _print_stats(meta)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
