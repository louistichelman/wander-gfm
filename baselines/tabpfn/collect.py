#!/usr/bin/env python3
"""Aggregate TabPFN Wander-split reports into a summary CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_WANDER_ROOT = Path(__file__).resolve().parents[2]
if str(_WANDER_ROOT) not in sys.path:
    sys.path.insert(0, str(_WANDER_ROOT))

from baselines.export_nc import add_nofeat_flags  # noqa: E402
from baselines.nodepfn.collect import (  # noqa: E402
    COLUMNS,
    SPLIT_COLUMNS,
    collect_splits,
)
from baselines.paths import WANDER_ROOT  # noqa: E402
from baselines.registry import nc_eval_specs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect TabPFN Wander-split results.")
    parser.add_argument("--output", type=str, default="")
    add_nofeat_flags(parser)
    args = parser.parse_args()

    folder = "nc-nofeat" if args.nofeat else "nc"
    results_root = WANDER_ROOT / "results" / "baselines" / "tabpfn" / folder
    output_path = Path(args.output) if args.output else results_root / "summary.csv"
    splits_path = output_path.with_name("splits.csv")
    specs = nc_eval_specs(nofeat=True if args.nofeat else False)

    summaries: list[dict] = []
    split_rows: list[dict] = []
    for spec in specs:
        summary, splits = collect_splits(results_root, spec)
        summaries.append(summary)
        split_rows.extend(splits)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(summaries)
    with splits_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SPLIT_COLUMNS)
        writer.writeheader()
        writer.writerows(split_rows)

    ok = sum(1 for row in summaries if row["status"] == "ok")
    skipped = sum(1 for row in summaries if row["status"] == "skipped")
    print(
        f"Wrote {output_path} and {splits_path} "
        f"({ok} ok, {skipped} skipped, {len(summaries)} total)"
    )


if __name__ == "__main__":
    main()
