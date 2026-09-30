#!/usr/bin/env python3
"""Aggregate AnyGraph NC reports into a summary CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_WANDER_ROOT = Path(__file__).resolve().parents[2]
if str(_WANDER_ROOT) not in sys.path:
    sys.path.insert(0, str(_WANDER_ROOT))

from baselines.export_nc import add_nofeat_flags  # noqa: E402
from baselines.paths import WANDER_ROOT  # noqa: E402
from baselines.registry import nc_eval_specs  # noqa: E402

COLUMNS = [
    "registry_key",
    "slug",
    "metric",
    "test_mean",
    "test_std",
    "val_mean",
    "val_std",
    "n_seeds",
    "report_path",
    "status",
]


def collect_one(results_root: Path, spec) -> dict:
    report_path = results_root / spec.slug / "report.json"
    rel = (
        str(report_path.relative_to(WANDER_ROOT))
        if report_path.is_relative_to(WANDER_ROOT)
        else str(report_path)
    )
    metric = "accuracy"
    if not report_path.is_file():
        return {
            "registry_key": spec.registry_key,
            "slug": spec.slug,
            "metric": metric,
            "test_mean": "",
            "test_std": "",
            "val_mean": "",
            "val_std": "",
            "n_seeds": 0,
            "report_path": rel,
            "status": "missing",
        }
    report = json.loads(report_path.read_text())
    test = report.get("metrics", {}).get("test", {})
    test_v = test.get("accuracy")
    test_std = test.get("accuracy_std")
    return {
        "registry_key": spec.registry_key,
        "slug": spec.slug,
        "metric": metric,
        "test_mean": "" if test_v is None else f"{float(test_v):.6f}",
        "test_std": "" if test_std is None else f"{float(test_std):.6f}",
        "val_mean": "",
        "val_std": "",
        "n_seeds": 10,
        "report_path": rel,
        "status": "ok" if test_v is not None else "empty",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect AnyGraph NC results.")
    parser.add_argument("--output", type=str, default="")
    add_nofeat_flags(parser)
    args = parser.parse_args()

    folder = "nc-nofeat" if args.nofeat else "nc"
    results_root = WANDER_ROOT / "results" / "baselines" / "anygraph" / folder
    output_path = (
        Path(args.output)
        if args.output
        else results_root / "summary.csv"
    )
    specs = nc_eval_specs(nofeat=True if args.nofeat else False)
    rows = [collect_one(results_root, spec) for spec in specs]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    ok = sum(1 for row in rows if row["status"] == "ok")
    print(f"[{folder}] Wrote {output_path} ({ok}/{len(rows)} datasets with results)")


if __name__ == "__main__":
    main()
