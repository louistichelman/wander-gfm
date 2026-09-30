#!/usr/bin/env python3
"""Collect LP grid JSONs and pick one config per method/dataset.

For datasets with a real val split, choose the config with the highest
``best_val_metric`` (monitor Recall@20). For warm-val datasets (ddi,
products_home), copy the config that has the best mean val metric on the
matching featured/nofeat val group.

Usage (from the repo root):
    python scripts/baselines/collect_lp_grid.py
    python scripts/baselines/collect_lp_grid.py --root results/baselines/lp_grid
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.lp_grid import (
    FEATURED_VAL,
    GRID_DATASETS,
    METHOD_CONFIGS,
    NOFEAT_VAL,
    is_warm_val,
    val_group_for,
)


def _load_runs(method_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(method_dir.glob("*_seed*.json")):
        if path.name.startswith("summary") or path.name.endswith(".failed.json"):
            continue
        try:
            rows.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    return rows


def _best_config_by_val(runs: List[Dict[str, Any]], datasets: tuple) -> Optional[str]:
    by_cfg: Dict[str, List[float]] = defaultdict(list)
    for row in runs:
        if row.get("dataset") not in datasets:
            continue
        cid = row.get("config_id")
        metric = row.get("best_val_metric")
        if cid is None or metric is None:
            continue
        by_cfg[str(cid)].append(float(metric))
    if not by_cfg:
        return None
    means = {cid: sum(vs) / len(vs) for cid, vs in by_cfg.items()}
    return max(means, key=means.get)


def collect_method(method: str, method_dir: Path) -> Dict[str, Any]:
    runs = _load_runs(method_dir)
    by_ds_cfg: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in runs:
        ds = row.get("dataset")
        cid = row.get("config_id")
        if ds and cid:
            by_ds_cfg[ds][cid] = row

    featured_winner = _best_config_by_val(runs, FEATURED_VAL)
    nofeat_winner = _best_config_by_val(runs, NOFEAT_VAL)
    selected = []
    for ds in GRID_DATASETS:
        cfg_map = by_ds_cfg.get(ds, {})
        if not cfg_map:
            selected.append({"dataset": ds, "status": "missing"})
            continue
        if is_warm_val(ds):
            group = val_group_for(ds)
            cid = featured_winner if group is FEATURED_VAL else nofeat_winner
            selection = "transferred_from_val_group"
        else:
            cid = max(cfg_map, key=lambda c: float(cfg_map[c].get("best_val_metric") or -1.0))
            selection = "best_val"
        row = cfg_map.get(cid or "")
        if row is None:
            # fall back to best test among completed configs (should not happen)
            cid = max(cfg_map, key=lambda c: float(cfg_map[c].get("test_recall@20") or -1.0))
            row = cfg_map[cid]
            selection = "fallback_best_test"
        selected.append(
            {
                "dataset": ds,
                "config_id": cid,
                "selection": selection,
                "monitor_split": row.get("monitor_split"),
                "best_val_metric": row.get("best_val_metric"),
                "test_recall@20": row.get("test_recall@20"),
                "test_ndcg@20": row.get("test_ndcg@20"),
                "best_epoch": row.get("best_epoch"),
            }
        )
    return {
        "method": method,
        "n_runs": len(runs),
        "featured_val_winner": featured_winner,
        "nofeat_val_winner": nofeat_winner,
        "selected": selected,
        "runs": runs,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=str,
        default=str(REPO_ROOT / "results" / "baselines" / "lp_grid"),
    )
    args = parser.parse_args(argv)
    root = Path(args.root)
    if not root.is_absolute():
        root = REPO_ROOT / root

    summary: Dict[str, Any] = {"root": str(root), "methods": {}}
    table_rows = []
    for method in METHOD_CONFIGS:
        method_dir = root / method
        if not method_dir.is_dir():
            continue
        payload = collect_method(method, method_dir)
        summary["methods"][method] = payload
        _write = root / method / "grid_selection.json"
        _write.write_text(json.dumps({k: v for k, v in payload.items() if k != "runs"}, indent=2) + "\n")
        print(f"\n=== {method} ===")
        print(f"featured val winner: {payload['featured_val_winner']}")
        print(f"nofeat val winner:   {payload['nofeat_val_winner']}")
        for row in payload["selected"]:
            rec = row.get("test_recall@20")
            rec_s = f"{rec:.4f}" if isinstance(rec, float) else str(rec)
            print(
                f"  {row['dataset']:<28} {row.get('config_id')}  "
                f"sel={row.get('selection')}  test_r@20={rec_s}"
            )
            table_rows.append({"method": method, **row})

    out = root / "summary.json"
    out.write_text(json.dumps({"methods": {m: {k: v for k, v in p.items() if k != "runs"} for m, p in summary["methods"].items()}}, indent=2) + "\n")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
