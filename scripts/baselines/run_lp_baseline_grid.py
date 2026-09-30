#!/usr/bin/env python3
"""Run the paper Hom-LP baseline grids (BUDDY 12 / NBFNet 18).

NBFNet trains for up to 40 epochs. Table 3 datasets by default.

Usage (from the repo root):
    python scripts/baselines/run_lp_baseline_grid.py --method buddy --datasets ANYGRAPH_CORA
    python scripts/baselines/run_lp_baseline_grid.py --method buddy --dry-run
    python scripts/baselines/run_lp_baseline_grid.py --method nbfnet --datasets ANYGRAPH_DDI
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.buddy.data import FEATURED_ANYGRAPH
from baselines.buddy.train import TrainConfig as BuddyConfig
from baselines.buddy.train import train_buddy
from baselines.lp_grid import (
    GRID_DATASETS,
    NBFNET_MAX_EPOCHS,
    SHARED,
    batches_per_epoch_for,
    configs_for,
    parse_dataset_list,
)
from baselines.nbfnet.data import bundle_to_nbfnet_graph
from baselines.nbfnet.train import TrainConfig as NbfConfig
from baselines.nbfnet.train import train_nbfnet
from data.dataset import DataSet, get_datasetargs


def _load_bundle(dataset: str, data_dir: str, pca_target_dim: int, seed: int):
    args = get_datasetargs(dataset)
    ds = DataSet(args, pca_target_dim=pca_target_dim)
    return ds.load(data_dir=data_dir, seed=seed)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _buddy_recipe(dataset: str) -> Dict[str, Any]:
    featured = dataset in FEATURED_ANYGRAPH
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


def _run_one(
    method: str,
    dataset: str,
    cfg_row: Dict[str, Any],
    *,
    data_dir: str,
    device: str,
    checkpoint_path: Path,
    resume: bool,
) -> Dict[str, Any]:
    seed = int(SHARED["seed"])
    bundle = _load_bundle(dataset, data_dir, int(SHARED["pca_target_dim"]), seed)
    graph = bundle_to_nbfnet_graph(bundle)
    bpe = batches_per_epoch_for(dataset, method)
    monitor = int(SHARED["max_monitor_queries"])
    n_neg = int(SHARED["num_negative"])

    if method == "buddy":
        recipe = _buddy_recipe(dataset)
        buddy_neg = int(cfg_row["num_negative"]) if "num_negative" in cfg_row else n_neg
        equal_w = bool(cfg_row.get("equal_pos_neg_weight", False))
        cfg = BuddyConfig(
            hidden_channels=int(cfg_row["hidden_channels"]),
            lr=float(cfg_row["lr"]),
            num_negative=buddy_neg,
            equal_pos_neg_weight=equal_w,
            batches_per_epoch=bpe,
            max_epochs=50,
            patience=8,
            max_monitor_queries=monitor,
            use_node_features=recipe["use_node_features"],
            train_node_embedding=recipe["train_node_embedding"],
            propagate_embeddings=recipe["propagate_embeddings"],
            sign_k=recipe["sign_k"],
            seed=seed,
            device=device,
        )
        result = train_buddy(graph, cfg, checkpoint_path=checkpoint_path, resume=resume)
        result["recipe"] = recipe
        result["train_config"] = {
            "hidden_channels": cfg.hidden_channels,
            "lr": cfg.lr,
            "sign_k": cfg.sign_k,
            "num_negative": cfg.num_negative,
            "equal_pos_neg_weight": cfg.equal_pos_neg_weight,
            "batches_per_epoch": cfg.batches_per_epoch,
            "max_epochs": cfg.max_epochs,
        }
        return result

    hidden_dims = tuple(int(x) for x in cfg_row["hidden_dims"])
    nbf_neg = int(cfg_row["num_negative"]) if "num_negative" in cfg_row else n_neg
    equal_w = bool(cfg_row.get("equal_pos_neg_weight", False))
    cfg = NbfConfig(
        input_dim=32,
        hidden_dims=hidden_dims,
        lr=float(cfg_row["lr"]),
        num_negative=nbf_neg,
        equal_pos_neg_weight=equal_w,
        batches_per_epoch=bpe,
        max_epochs=NBFNET_MAX_EPOCHS,
        max_monitor_queries=monitor,
        seed=seed,
        device=device,
    )
    result = train_nbfnet(graph, cfg, checkpoint_path=checkpoint_path, resume=resume)
    result["train_config"] = {
        "hidden_dims": list(cfg.hidden_dims),
        "lr": cfg.lr,
        "num_negative": cfg.num_negative,
        "equal_pos_neg_weight": cfg.equal_pos_neg_weight,
        "batches_per_epoch": cfg.batches_per_epoch,
        "max_epochs": cfg.max_epochs,
        "max_monitor_queries": cfg.max_monitor_queries,
    }
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", type=str, required=True, choices=("buddy", "nbfnet"))
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(GRID_DATASETS),
        help="Comma- or space-separated registry keys (ANYGRAPH_CORA ANYGRAPH_DDI, ...)",
    )
    parser.add_argument("--data_dir", type=str, default="./raw_data")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Default: results/baselines/lp_grid/<method>/",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="",
        help="Default: checkpoints/baselines/lp_grid/<method>/",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--skip_finished",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--neg_grid",
        action="store_true",
        help=(
            "Deprecated no-op: NBFNet/BUDDY already run the paper "
            "architecture × negative/weight grid (18 / 12)."
        ),
    )
    parser.add_argument(
        "--nbfnet_neg_grid",
        action="store_true",
        help="Alias for --neg_grid with --method nbfnet.",
    )
    parser.add_argument(
        "--buddy_neg_grid",
        action="store_true",
        help="Alias for --neg_grid with --method buddy.",
    )
    args = parser.parse_args(argv)

    method = args.method
    neg_grid = bool(args.neg_grid or args.nbfnet_neg_grid or args.buddy_neg_grid)
    if args.nbfnet_neg_grid and method != "nbfnet":
        parser.error("--nbfnet_neg_grid requires --method nbfnet")
    if args.buddy_neg_grid and method != "buddy":
        parser.error("--buddy_neg_grid requires --method buddy")
    if neg_grid and method not in ("nbfnet", "buddy"):
        parser.error("--neg_grid requires --method nbfnet or buddy")
    del neg_grid
    datasets = parse_dataset_list(args.datasets)
    configs = configs_for(method)
    default_subdir = method
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else REPO_ROOT / "results" / "baselines" / "lp_grid" / default_subdir
    )
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    ckpt_root = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else REPO_ROOT / "checkpoints" / "baselines" / "lp_grid" / default_subdir
    )
    if not ckpt_root.is_absolute():
        ckpt_root = REPO_ROOT / ckpt_root
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    print(f"Method:      {method}")
    print(f"Datasets:    {datasets}")
    print(f"Configs:     {[c['id'] for c in configs]}")
    print(f"Output:      {output_dir}")
    print(f"Checkpoints: {ckpt_root}")
    if args.dry_run:
        return 0

    for dataset in datasets:
        for cfg_row in configs:
            cid = cfg_row["id"]
            stem = f"{dataset}_{cid}_seed{SHARED['seed']}"
            out_path = output_dir / f"{stem}.json"
            print(f"\n=== {method} {stem} ===", flush=True)
            if args.skip_finished and out_path.is_file():
                print(f"Skip (already finished): {out_path}", flush=True)
                continue
            ckpt_path = ckpt_root / f"{stem}_best.pt"
            if not args.resume and ckpt_path.is_file():
                ckpt_path.unlink()
            try:
                result = _run_one(
                    method,
                    dataset,
                    cfg_row,
                    data_dir=args.data_dir,
                    device=args.device,
                    checkpoint_path=ckpt_path,
                    resume=args.resume,
                )
                payload = {
                    "method": method,
                    "dataset": dataset,
                    "config_id": cid,
                    "seed": SHARED["seed"],
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "grid": cfg_row,
                    **result,
                }
                _write_json(out_path, payload)
                print(f"Wrote {out_path}", flush=True)
            except Exception as exc:
                fail_path = output_dir / f"{stem}.failed.json"
                _write_json(
                    fail_path,
                    {
                        "method": method,
                        "dataset": dataset,
                        "config_id": cid,
                        "error": f"{type(exc).__name__}: {exc}",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
                print(f"FAILED {stem}: {exc}", flush=True)
                raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
