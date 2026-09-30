#!/usr/bin/env python3
"""Pick the best NC finetune checkpoint for one dataset (highest val score).

Prints ``ckpt_path\\tscore\\trun_name`` for the winning LR, or exits 2 if none
of the grid runs have a usable val score + ``model_seed*_best.pt``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def _score_from_best_val(run_dir: Path) -> Optional[float]:
    path = run_dir / "best_val.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    score = payload.get("best_val_score")
    return None if score is None else float(score)


def _score_from_eval_epochs(run_dir: Path) -> Optional[float]:
    epoch_dir = run_dir / "eval_epochs"
    if not epoch_dir.is_dir():
        return None
    best: Optional[float] = None
    for path in epoch_dir.glob("epoch_*.json"):
        if path.name.endswith("_test.json"):
            continue
        payload = json.loads(path.read_text())
        score = payload.get("val_score")
        if score is None:
            continue
        value = float(score)
        if best is None or value > best:
            best = value
    return best


def score_for_run(run_dir: Path) -> Optional[float]:
    return _score_from_best_val(run_dir) or _score_from_eval_epochs(run_dir)


def find_best_pt(run_dir: Path, seed: int) -> Optional[Path]:
    preferred = run_dir / f"model_seed{seed}_best.pt"
    if preferred.is_file():
        return preferred
    matches = sorted(run_dir.glob("model_seed*_best.pt"))
    return matches[0] if matches else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print the best NC finetune checkpoint for one dataset."
    )
    parser.add_argument("dataset", help="Registry key, e.g. CORA")
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Root checkpoint directory (default: checkpoints).",
    )
    parser.add_argument(
        "--run-prefix",
        default="ft",
        help="Run-name prefix (default: ft). Looks for {prefix}_{DATASET}_nc_lr*.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dataset = args.dataset
    pattern = f"{args.run_prefix}_{dataset}_nc_lr*"
    candidates = []
    for run_dir in sorted(args.checkpoints_dir.glob(pattern)):
        if not run_dir.is_dir():
            continue
        ckpt = find_best_pt(run_dir, args.seed)
        score = score_for_run(run_dir)
        if ckpt is None or score is None:
            continue
        candidates.append((score, ckpt.resolve(), run_dir.name))

    if not candidates:
        print(
            f"ERROR: no completed NC finetune runs for {dataset} "
            f"under {args.checkpoints_dir}/{pattern}",
            file=sys.stderr,
        )
        return 2

    candidates.sort(key=lambda row: row[0], reverse=True)
    score, ckpt, run_name = candidates[0]
    print(f"{ckpt}\t{score:.6f}\t{run_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
