#!/usr/bin/env python3
"""Build per-source sampled-recall caches next to UniLP split files.

Usage (from the repo root):
  python scripts/eval/build_unilp_sampled_recall.py \\
    --unilp_data_dir third_party/context_LP/data \\
    --datasets USAir,NS,PB --runs 7 --num_neg 100
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from baselines.paths import resolve_unilp_data_root
from data.datasets.unilp_lp import (
    UNILP_LARGE_DATASETS,
    UNILP_SMALL_DATASETS,
    load_unilp_split,
    resolve_unilp_data_dir,
)
from data.sampled_recall import (
    DEFAULT_NUM_NEG,
    load_or_build_sampled_recall_cache,
    sampled_recall_cache_path,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--unilp_data_dir",
        default=str(resolve_unilp_data_root()),
    )
    p.add_argument(
        "--datasets",
        default=",".join(UNILP_SMALL_DATASETS),
        help="Comma-separated UniLP folder names. "
        f"Large Table 1 graphs: {','.join(UNILP_LARGE_DATASETS)}.",
    )
    p.add_argument("--runs", type=int, default=7)
    p.add_argument(
        "--run_ids",
        default="",
        help="Comma-separated run indices (e.g. 1,2). Overrides --runs.",
    )
    p.add_argument("--num_neg", type=int, default=DEFAULT_NUM_NEG)
    p.add_argument("--splits", default="valid,test")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_unilp_data_dir(args.unilp_data_dir)
    datasets = [n.strip() for n in args.datasets.split(",") if n.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    print(f"UniLP data: {root}")
    if str(args.run_ids).strip():
        run_ids = [int(x) for x in str(args.run_ids).split(",") if x.strip()]
    else:
        run_ids = list(range(int(args.runs)))
    print(f"datasets={datasets} runs={run_ids} num_neg={args.num_neg}")
    for name in datasets:
        for run in run_ids:
            data, split_edge = load_unilp_split(name, root, run=run)
            n = int(maybe_num_nodes(data.edge_index, data.num_nodes))
            folder = root / name
            for split in splits:
                cache = load_or_build_sampled_recall_cache(
                    folder,
                    split_edge,
                    n,
                    run=run,
                    split=split,
                    num_neg=int(args.num_neg),
                )
                path = sampled_recall_cache_path(folder, run, split, args.num_neg)
                print(
                    f"  {name} run={run} {split}: "
                    f"S={cache['sources'].numel()} "
                    f"P={cache['pos_tails'].numel()} "
                    f"N={cache['neg_tails'].numel()} -> {path}"
                )


if __name__ == "__main__":
    main()
