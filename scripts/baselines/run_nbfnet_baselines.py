#!/usr/bin/env python3
"""Train/evaluate NBFNet LP baseline on AnyGraph datasets.

Usage (from the repo root):
    python scripts/baselines/run_nbfnet_baselines.py --datasets ANYGRAPH_CORA --seeds 0
    python scripts/baselines/run_nbfnet_baselines.py --datasets ANYGRAPH_CORA --max_epochs 5 --dry-run
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

from baselines.lp_grid import parse_dataset_list
from baselines.nbfnet.data import bundle_to_nbfnet_graph
from baselines.nbfnet.train import TrainConfig, train_nbfnet
from data.dataset import DataSet, get_datasetargs

DEFAULT_OUTPUT = REPO_ROOT / "results" / "baselines" / "nbfnet"
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "baselines" / "nbfnet"


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
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, {"runs": rows, "n": len(rows)})

    # Flatten CSV
    keys = sorted({k for r in rows for k in r.keys() if k != "history"})
    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in keys})


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        type=str,
        default="ANYGRAPH_CORA",
        help="Comma- or space-separated registry keys (ANYGRAPH_CORA ANYGRAPH_DDI, ...)",
    )
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--data_dir", type=str, default="./raw_data")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="Directory for {dataset}_seed{N}_best.pt weights (default: checkpoints/baselines/nbfnet).",
    )
    parser.add_argument("--pca_target_dim", type=int, default=64)
    parser.add_argument("--input_dim", type=int, default=32)
    parser.add_argument("--hidden_dims", type=str, default="32,32,32,32,32,32")
    parser.add_argument("--num_negative", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="Eval query batch size (default: min(4, batch_size)).",
    )
    parser.add_argument(
        "--batches_per_epoch",
        type=int,
        default=None,
        help=(
            "If set and train edges > batches_per_epoch * batch_size, each epoch "
            "runs exactly this many random batches; otherwise a full train pass."
        ),
    )
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--max_eval_queries",
        type=int,
        default=None,
        help="Eval query/source cap. Default: no cap (full test split).",
    )
    parser.add_argument(
        "--max_monitor_queries",
        type=int,
        default=None,
        help="Source cap for train-time monitor (final test stays uncapped unless --max_eval_queries).",
    )
    parser.add_argument("--recall_k", type=int, default=20)
    parser.add_argument(
        "--use_node_features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inject node features when available (default: on).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from {dataset}_seed{N}_best.pt in checkpoint_dir when present (default: on).",
    )
    parser.add_argument(
        "--skip_finished",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip datasets that already have a success JSON in output_dir (default: on).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    datasets = parse_dataset_list(args.datasets)
    seeds = [int(s) for s in _parse_list(args.seeds)]
    hidden_dims = tuple(int(x) for x in _parse_list(args.hidden_dims))
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
                eval_bs = (
                    args.eval_batch_size
                    if args.eval_batch_size is not None
                    else min(4, int(args.batch_size))
                )
                max_eval_queries = args.max_eval_queries
                cfg = TrainConfig(
                    input_dim=args.input_dim,
                    hidden_dims=hidden_dims,
                    num_negative=args.num_negative,
                    batch_size=args.batch_size,
                    eval_batch_size=eval_bs,
                    batches_per_epoch=args.batches_per_epoch,
                    lr=args.lr,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    use_node_features=args.use_node_features,
                    max_eval_queries=max_eval_queries,
                    max_monitor_queries=args.max_monitor_queries,
                    recall_k=args.recall_k,
                    seed=seed,
                    device=args.device,
                )
                ckpt_path = checkpoint_dir / f"{stem}_best.pt"
                if not args.resume and ckpt_path.is_file():
                    ckpt_path.unlink()
                    print(f"Removed checkpoint for fresh start: {ckpt_path}", flush=True)
                result = train_nbfnet(
                    graph, cfg, checkpoint_path=ckpt_path, resume=args.resume
                )
                payload = {
                    "dataset": dataset,
                    "seed": seed,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "pca_target_dim": args.pca_target_dim,
                    "config": {
                        "input_dim": cfg.input_dim,
                        "hidden_dims": list(cfg.hidden_dims),
                        "num_negative": cfg.num_negative,
                        "batch_size": cfg.batch_size,
                        "eval_batch_size": cfg.eval_batch_size,
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
