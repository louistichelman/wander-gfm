#!/usr/bin/env python3
"""BFS-subsample Pokec Regions to ~100k nodes and print class / split counts.

Protocol: start at a random train node of the third-largest class, then keep
the first 100k nodes in BFS order (induced subgraph, original RL masks).

Usage (from the repo root):
    python scripts/data/subsample_pokec_regions.py
    python scripts/data/subsample_pokec_regions.py --data_dir raw_data --force
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from data.bfs_subsample import (  # noqa: E402
    class_split_rows,
    node_class_labels,
)
from data.datasets import pokec_regions_100k  # noqa: E402


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _print_full_ranking(ranked: torch.Tensor, sizes: torch.Tensor, class_rank: int) -> None:
    print("Full-graph class sizes (top 10):")
    k = min(10, int(ranked.numel()))
    for i in range(k):
        cid = int(ranked[i].item())
        n = int(sizes[cid].item())
        mark = "  <-- seed class" if i == class_rank else ""
        print(f"  rank {i + 1:2d}  class {cid:4d}  n={_fmt_int(n)}{mark}")
    if ranked.numel() > k:
        print(f"  ... {ranked.numel() - k} more classes")


def _print_table(rows) -> None:
    print()
    print(
        f"{'orig_class':>12} {'n':>10} {'train':>10} {'val':>10} {'test':>10} {'none':>8}"
    )
    print("-" * 64)
    tot = {"n": 0, "train": 0, "val": 0, "test": 0, "none": 0}
    for row in rows:
        label = "unlabeled" if row.orig_class is None else str(row.orig_class)
        print(
            f"{label:>12} {_fmt_int(row.n):>10} {_fmt_int(row.train):>10} "
            f"{_fmt_int(row.val):>10} {_fmt_int(row.test):>10} {_fmt_int(row.none):>8}"
        )
        tot["n"] += row.n
        tot["train"] += row.train
        tot["val"] += row.val
        tot["test"] += row.test
        tot["none"] += row.none
    print("-" * 64)
    print(
        f"{'total':>12} {_fmt_int(tot['n']):>10} {_fmt_int(tot['train']):>10} "
        f"{_fmt_int(tot['val']):>10} {_fmt_int(tot['test']):>10} {_fmt_int(tot['none']):>8}"
    )


def _write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["orig_class", "n", "train", "val", "test", "none"]
        )
        w.writeheader()
        for row in rows:
            w.writerow(
                {
                    "orig_class": "" if row.orig_class is None else row.orig_class,
                    "n": row.n,
                    "train": row.train,
                    "val": row.val,
                    "test": row.test,
                    "none": row.none,
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(REPO_ROOT / "raw_data"),
        help="Root data directory (default: Wander/raw_data)",
    )
    parser.add_argument("--seed", type=int, default=pokec_regions_100k.SEED)
    parser.add_argument("--target_nodes", type=int, default=pokec_regions_100k.TARGET_NODES)
    parser.add_argument(
        "--class_rank",
        type=int,
        default=pokec_regions_100k.CLASS_RANK,
        help="0 = largest class; 2 = third-largest (default).",
    )
    parser.add_argument(
        "--keep_top_classes",
        type=int,
        default=None,
        help="Keep labels/splits only for the K largest classes; rest unlabeled.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the cached subsample even if it already exists.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Optional CSV path for the class × split table.",
    )
    args = parser.parse_args()

    cache = pokec_regions_100k.cache_path(
        args.data_dir,
        seed=args.seed,
        target_nodes=args.target_nodes,
        class_rank=args.class_rank,
        keep_top_classes=args.keep_top_classes,
    )
    print(
        f"Protocol: random train seed in class-rank {args.class_rank} "
        f"(0=largest), BFS to {args.target_nodes:,} nodes, seed={args.seed}"
        + (
            f", keep top {args.keep_top_classes} classes labeled"
            if args.keep_top_classes is not None
            else ""
        )
    )
    print(f"Cache: {cache}")
    data = pokec_regions_100k.load_base_dataset(
        args.data_dir,
        seed=args.seed,
        force=args.force,
        target_nodes=args.target_nodes,
        class_rank=args.class_rank,
        keep_top_classes=args.keep_top_classes,
    )

    ranked = data.full_ranked_classes
    sizes = data.full_class_sizes
    n_sub = int(data.num_nodes)
    n_edges = int(data.edge_index.size(1) // 2)
    avg_deg = (2.0 * n_edges / n_sub) if n_sub else 0.0
    n_classes = int(data.orig_class_ids.numel())

    print()
    _print_full_ranking(ranked, sizes, int(data.bfs_class_rank))
    print()
    print(f"Seed node (original id): {int(data.bfs_seed_node)}")
    print(
        f"Seed class: {int(data.bfs_seed_class)} "
        f"(full-graph size {int(sizes[int(data.bfs_seed_class)].item()):,})"
    )
    print(
        f"Subsample: {n_sub:,} nodes, {n_edges:,} undirected edges, "
        f"avg degree {avg_deg:.2f}, max BFS hop {int(data.bfs_max_hop)}"
    )
    print(
        f"Hit {args.target_nodes:,} budget: {bool(data.bfs_hit_budget)}; "
        f"classes labeled: {n_classes}"
        + (
            f" (top {int(data.keep_top_classes)}; "
            f"{int(data.dropped_class_ids.numel())} dropped to unlabeled)"
            if getattr(data, "keep_top_classes", None) is not None
            else f" / {int(ranked.numel())} on full graph"
        )
    )

    labels, labeled = node_class_labels(data.orig_y)
    rows = class_split_rows(
        labels, labeled, data.train_mask, data.val_mask, data.test_mask
    )
    _print_table(rows)

    csv_path = Path(args.output_csv) if args.output_csv else cache.with_suffix(".csv")
    _write_csv(csv_path, rows)
    print(f"\nWrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
