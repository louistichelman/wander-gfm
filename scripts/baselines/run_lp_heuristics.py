#!/usr/bin/env python3
"""Evaluate CN / RA / PA on homogeneous AnyGraph LP graphs.

Usage (from the repo root):
    python scripts/baselines/run_lp_heuristics.py
    python scripts/baselines/run_lp_heuristics.py --datasets ANYGRAPH_CORA --dry-run

Default set is the 10 homogeneous Table 3 reporting-set graphs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.heuristics import HEURISTIC_METHODS, evaluate_all_heuristics
from baselines.lp_grid import HOMOGENEOUS_DATASETS, parse_dataset_list
from baselines.nbfnet.data import bundle_to_nbfnet_graph
from data.dataset import DataSet, get_datasetargs

DEFAULT_OUTPUT = REPO_ROOT / "results" / "baselines" / "heuristics"


def _parse_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _load_bundle(dataset: str, data_dir: str, pca_target_dim: int, seed: int):
    args = get_datasetargs(dataset)
    ds = DataSet(args, pca_target_dim=pca_target_dim)
    return ds.load(data_dir=data_dir, seed=seed)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _update_summary(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.glob("*_seed*.json")):
        if path.name.startswith("summary") or path.name.endswith(".failed.json"):
            continue
        if "_cn_" in path.name or "_ra_" in path.name or "_pa_" in path.name:
            try:
                rows.append(json.loads(path.read_text()))
            except json.JSONDecodeError:
                continue
    if not rows:
        return
    _write_json(output_dir / "summary.json", {"runs": rows, "n": len(rows)})
    keys = sorted({k for r in rows for k in r.keys() if k != "history"})
    with (output_dir / "summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in keys})


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(HOMOGENEOUS_DATASETS),
        help="Comma- or space-separated registry keys (ANYGRAPH_CORA ANYGRAPH_DDI, ...)",
    )
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--data_dir", type=str, default="./raw_data")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--pca_target_dim", type=int, default=64)
    parser.add_argument("--recall_k", type=int, default=20)
    parser.add_argument("--max_eval_queries", type=int, default=None)
    parser.add_argument("--source_batch_size", type=int, default=64)
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(HEURISTIC_METHODS),
        help="Comma-separated subset of cn,ra,pa",
    )
    parser.add_argument(
        "--skip_finished",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    datasets = parse_dataset_list(args.datasets)
    seeds = [int(s) for s in _parse_list(args.seeds)]
    methods = [m.lower() for m in _parse_list(args.methods)]
    unknown = [m for m in methods if m not in HEURISTIC_METHODS]
    if unknown:
        raise SystemExit(f"unknown methods {unknown}; expected {list(HEURISTIC_METHODS)}")
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Datasets:  {datasets}")
    print(f"Methods:   {methods}")
    print(f"Seeds:     {seeds}")
    print(f"Output:    {output_dir}")
    if args.dry_run:
        return 0

    for dataset in datasets:
        for seed in seeds:
            method_paths = {
                method: output_dir / f"{dataset}_{method}_seed{seed}.json"
                for method in methods
            }
            if args.skip_finished and all(p.is_file() for p in method_paths.values()):
                print(f"Skip (already finished): {dataset} seed {seed}", flush=True)
                continue
            print(f"\n=== {dataset} seed {seed} ===", flush=True)
            try:
                bundle = _load_bundle(dataset, args.data_dir, args.pca_target_dim, seed)
                graph = bundle_to_nbfnet_graph(bundle)
                if graph.is_bipartite:
                    print(
                        f"WARNING: {dataset} loaded as bipartite "
                        f"(offset={graph.candidate_offset}); 1-hop CN/RA is 0 "
                        "on the item pool. Continuing anyway.",
                        flush=True,
                    )
                results = evaluate_all_heuristics(
                    graph,
                    methods=methods,
                    recall_k=args.recall_k,
                    max_queries=args.max_eval_queries,
                    seed=seed,
                    source_batch_size=args.source_batch_size,
                )
                timestamp = datetime.now(timezone.utc).isoformat()
                for method, metrics in results.items():
                    payload = {
                        **metrics,
                        "dataset": dataset,
                        "seed": seed,
                        "method": method,
                        "timestamp_utc": timestamp,
                        "pca_target_dim": args.pca_target_dim,
                        "is_bipartite": graph.is_bipartite,
                        "candidate_offset": graph.candidate_offset,
                        "num_nodes": int(graph.num_nodes),
                        "num_train_triples": int(graph.train_triples.shape[0]),
                        "num_val_triples": int(graph.val_triples.shape[0]),
                        "num_test_triples": int(graph.test_triples.shape[0]),
                        "recall_k": args.recall_k,
                        "max_eval_queries": args.max_eval_queries,
                    }
                    out_path = method_paths[method]
                    _write_json(out_path, payload)
                    rec = metrics.get(f"test/recall@{args.recall_k}")
                    print(
                        f"  {method}: test/recall@{args.recall_k}={rec:.4f} "
                        f"-> {out_path}",
                        flush=True,
                    )
            except Exception as exc:
                fail_path = output_dir / f"{dataset}_seed{seed}.failed.json"
                _write_json(
                    fail_path,
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "error": f"{type(exc).__name__}: {exc}",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
                print(f"FAILED {dataset} seed {seed}: {exc}", flush=True)
                raise

    _update_summary(output_dir)
    print(f"\nSummary: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
