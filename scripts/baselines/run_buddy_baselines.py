#!/usr/bin/env python3
"""Train/evaluate BUDDY LP baseline on the AnyGraph reporting set.

Usage (from the repo root):
    python scripts/baselines/run_buddy_baselines.py --datasets ANYGRAPH_CORA --seeds 0
    python scripts/baselines/run_buddy_baselines.py --dry-run
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

from baselines.buddy.data import FEATURED_ANYGRAPH, bundle_to_nbfnet_graph
from baselines.buddy.train import TrainConfig, train_buddy
from baselines.lp_grid import parse_dataset_list
from data.dataset import DataSet, get_datasetargs

REPORTING_DATASETS = (
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

DEFAULT_OUTPUT = REPO_ROOT / "results" / "baselines" / "buddy"
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "baselines" / "buddy"


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


def _recipe_for(dataset: str, use_node_features_flag: bool) -> dict:
    featured = dataset in FEATURED_ANYGRAPH and use_node_features_flag
    if featured:
        return {
            "use_node_features": True,
            "sign_k": 0,
            "train_node_embedding": False,
            "propagate_embeddings": False,
        }
    return {
        "use_node_features": False,
        "sign_k": 2,
        "train_node_embedding": True,
        "propagate_embeddings": True,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(REPORTING_DATASETS),
        help="Comma- or space-separated registry keys (ANYGRAPH_CORA ANYGRAPH_DDI, ...)",
    )
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--data_dir", type=str, default="./raw_data")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="Directory for {dataset}_seed{N}_best.pt weights.",
    )
    parser.add_argument("--pca_target_dim", type=int, default=64)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--max_hash_hops", type=int, default=2)
    parser.add_argument("--num_negative", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--eval_pair_batch_size", type=int, default=250000)
    parser.add_argument("--batches_per_epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max_eval_queries", type=int, default=None)
    parser.add_argument(
        "--max_monitor_queries",
        type=int,
        default=512,
        help="Source cap for train-time val (final test stays uncapped unless --max_eval_queries).",
    )
    parser.add_argument("--recall_k", type=int, default=20)
    parser.add_argument(
        "--use_node_features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use node features on featured reporting-set graphs (default: on).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from {dataset}_seed{N}_best.pt when present (default: on).",
    )
    parser.add_argument(
        "--skip_finished",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip datasets that already have a success JSON (default: on).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    datasets = parse_dataset_list(args.datasets)
    seeds = [int(s) for s in _parse_list(args.seeds)]
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = REPO_ROOT / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Datasets:     {datasets}")
    print(f"Seeds:        {seeds}")
    print(f"Output:       {output_dir}")
    print(f"Checkpoints:  {checkpoint_dir}")
    if args.dry_run:
        return 0

    for dataset in datasets:
        recipe = _recipe_for(dataset, args.use_node_features)
        for seed in seeds:
            stem = f"{dataset}_seed{seed}"
            out_path = output_dir / f"{stem}.json"
            print(f"\n=== {stem} ===", flush=True)
            if args.skip_finished and out_path.is_file():
                print(f"Skip (already finished): {out_path}", flush=True)
                continue
            try:
                bundle = _load_bundle(dataset, args.data_dir, args.pca_target_dim, seed)
                graph = bundle_to_nbfnet_graph(bundle)
                cfg = TrainConfig(
                    hidden_channels=args.hidden_channels,
                    max_hash_hops=args.max_hash_hops,
                    num_negative=args.num_negative,
                    batch_size=args.batch_size,
                    eval_pair_batch_size=args.eval_pair_batch_size,
                    batches_per_epoch=args.batches_per_epoch,
                    lr=args.lr,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    use_node_features=recipe["use_node_features"],
                    train_node_embedding=recipe["train_node_embedding"],
                    propagate_embeddings=recipe["propagate_embeddings"],
                    sign_k=recipe["sign_k"],
                    max_eval_queries=args.max_eval_queries,
                    max_monitor_queries=args.max_monitor_queries,
                    recall_k=args.recall_k,
                    seed=seed,
                    device=args.device,
                )
                ckpt_path = checkpoint_dir / f"{stem}_best.pt"
                if not args.resume and ckpt_path.is_file():
                    ckpt_path.unlink()
                    print(f"Removed checkpoint for fresh start: {ckpt_path}", flush=True)
                result = train_buddy(graph, cfg, checkpoint_path=ckpt_path, resume=args.resume)
                payload = {
                    "dataset": dataset,
                    "seed": seed,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "pca_target_dim": args.pca_target_dim,
                    "recipe": recipe,
                    "config": {
                        "hidden_channels": cfg.hidden_channels,
                        "max_hash_hops": cfg.max_hash_hops,
                        "sign_k": cfg.sign_k,
                        "num_negative": cfg.num_negative,
                        "batch_size": cfg.batch_size,
                        "eval_pair_batch_size": cfg.eval_pair_batch_size,
                        "batches_per_epoch": cfg.batches_per_epoch,
                        "lr": cfg.lr,
                        "max_epochs": cfg.max_epochs,
                        "patience": cfg.patience,
                        "max_eval_queries": cfg.max_eval_queries,
                        "max_monitor_queries": cfg.max_monitor_queries,
                        "recall_k": cfg.recall_k,
                        "use_node_features_flag": args.use_node_features,
                    },
                    **result,
                }
                _write_json(out_path, payload)
                print(f"Wrote {out_path}", flush=True)
            except Exception as exc:
                fail_path = output_dir / f"{stem}.failed.json"
                _write_json(
                    fail_path,
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "error": f"{type(exc).__name__}: {exc}",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
                print(f"FAILED {stem}: {exc}", flush=True)
                raise

    _update_summary(output_dir)
    print(f"\nSummary: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
