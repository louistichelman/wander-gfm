#!/usr/bin/env python3
"""Aggregate GraphAny Wander-split reports into a summary CSV.

Multi-split datasets (official masks or GraphAny 0..4) are averaged over the
protocol splits.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

_WANDER_ROOT = Path(__file__).resolve().parents[2]
if str(_WANDER_ROOT) not in sys.path:
    sys.path.insert(0, str(_WANDER_ROOT))

from baselines.export_nc import add_nofeat_flags  # noqa: E402
from baselines.paths import WANDER_ROOT  # noqa: E402
from baselines.registry import nc_eval_specs  # noqa: E402
from data.nc_splits import (  # noqa: E402
    SPLIT_DIR_PREFIX,
    eval_nc_split_indices,
    n_eval_nc_splits,
)

COLUMNS = [
    "registry_key",
    "slug",
    "metric",
    "test_mean",
    "test_std",
    "val_mean",
    "val_std",
    "n_seeds",
    "n_splits",
    "n_split_reports",
    "checkpoint",
    "report_path",
    "status",
]

SPLIT_COLUMNS = [
    "registry_key",
    "slug",
    "split_index",
    "metric",
    "test_mean",
    "test_std",
    "val_mean",
    "val_std",
    "n_seeds",
    "checkpoint",
    "report_path",
    "status",
]


def _metric_key(spec) -> str:
    return "roc-auc" if spec.is_binary else "accuracy"


def _summarize_scores(scores: list[float]) -> tuple[float | None, float | None]:
    if not scores:
        return None, None
    if len(scores) == 1:
        return scores[0], 0.0
    return statistics.mean(scores), statistics.pstdev(scores)


def _report_relpath(report_path: Path) -> str:
    try:
        return str(report_path.relative_to(WANDER_ROOT))
    except ValueError:
        return str(report_path)


def _load_report(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _split_report_path(results_root: Path, spec, split_index: int) -> Path:
    slug_root = results_root / spec.slug
    if n_eval_nc_splits(spec.registry_key) <= 1:
        return slug_root / "report.json"
    return slug_root / f"{SPLIT_DIR_PREFIX}{int(split_index)}" / "report.json"


def _row_from_report(spec, report: dict, report_path: Path, metric_label: str) -> dict:
    metric_key = _metric_key(spec)
    test = report.get("metrics", {}).get("test", {})
    val = report.get("metrics", {}).get("val", {})
    if spec.is_binary and test.get("roc-auc") is None:
        metric_key = "accuracy"
        metric_label = "accuracy"
    test_v = test.get(metric_key)
    val_v = val.get(metric_key)
    checkpoint = str(report.get("checkpoint", ""))
    return {
        "registry_key": spec.registry_key,
        "slug": spec.slug,
        "metric": metric_label if test_v is None else metric_key,
        "test_mean": "" if test_v is None else f"{float(test_v):.6f}",
        "test_std": "0.000000" if test_v is not None else "",
        "val_mean": "" if val_v is None else f"{float(val_v):.6f}",
        "val_std": "0.000000" if val_v is not None else "",
        "n_seeds": 1 if test_v is not None else 0,
        "checkpoint": checkpoint,
        "report_path": _report_relpath(report_path),
        "status": "ok" if test_v is not None else "empty",
        "_test_mean": None if test_v is None else float(test_v),
        "_val_mean": None if val_v is None else float(val_v),
    }


def _empty_split_row(spec, split_index: int, report_path: Path, metric_label: str) -> dict:
    return {
        "registry_key": spec.registry_key,
        "slug": spec.slug,
        "split_index": split_index,
        "metric": metric_label,
        "test_mean": "",
        "test_std": "",
        "val_mean": "",
        "val_std": "",
        "n_seeds": 0,
        "checkpoint": "",
        "report_path": _report_relpath(report_path),
        "status": "missing",
    }


def collect_splits(results_root: Path, spec) -> tuple[dict, list[dict]]:
    metric_label = _metric_key(spec)
    n_splits = n_eval_nc_splits(spec.registry_key)
    indices = eval_nc_split_indices(spec.registry_key)
    if n_splits <= 1:
        report_path = _split_report_path(results_root, spec, 0)
        report = _load_report(report_path)
        if report is None:
            summary = {
                "registry_key": spec.registry_key,
                "slug": spec.slug,
                "metric": metric_label,
                "test_mean": "",
                "test_std": "",
                "val_mean": "",
                "val_std": "",
                "n_seeds": 0,
                "n_splits": n_splits,
                "n_split_reports": 0,
                "checkpoint": "",
                "report_path": _report_relpath(report_path),
                "status": "missing",
            }
            return summary, [_empty_split_row(spec, 0, report_path, metric_label)]
        row = _row_from_report(spec, report, report_path, metric_label)
        summary = {
            **{k: v for k, v in row.items() if not k.startswith("_")},
            "n_splits": n_splits,
            "n_split_reports": 1 if row["status"] == "ok" else 0,
        }
        split_row = {
            **{k: v for k, v in row.items() if not k.startswith("_")},
            "split_index": 0,
        }
        return summary, [split_row]

    split_rows: list[dict] = []
    test_means: list[float] = []
    val_means: list[float] = []
    n_seeds_total = 0
    ok_paths: list[str] = []
    checkpoint = ""
    for split_index in indices:
        report_path = _split_report_path(results_root, spec, split_index)
        report = _load_report(report_path)
        if report is None:
            split_rows.append(_empty_split_row(spec, split_index, report_path, metric_label))
            continue
        row = _row_from_report(spec, report, report_path, metric_label)
        split_rows.append(
            {
                **{k: v for k, v in row.items() if not k.startswith("_")},
                "split_index": split_index,
            }
        )
        checkpoint = checkpoint or str(row.get("checkpoint", ""))
        if row["_test_mean"] is not None:
            test_means.append(row["_test_mean"])
            n_seeds_total += int(row["n_seeds"])
            ok_paths.append(row["report_path"])
        if row["_val_mean"] is not None:
            val_means.append(row["_val_mean"])

    test_mean, test_std = _summarize_scores(test_means)
    val_mean, val_std = _summarize_scores(val_means)
    metric = next(
        (r["metric"] for r in split_rows if r["status"] == "ok"),
        metric_label,
    )
    if not test_means:
        status = "missing"
    elif len(test_means) < n_splits:
        status = "partial"
    else:
        status = "ok"
    summary = {
        "registry_key": spec.registry_key,
        "slug": spec.slug,
        "metric": metric,
        "test_mean": "" if test_mean is None else f"{test_mean:.6f}",
        "test_std": "" if test_std is None else f"{test_std:.6f}",
        "val_mean": "" if val_mean is None else f"{val_mean:.6f}",
        "val_std": "" if val_std is None else f"{val_std:.6f}",
        "n_seeds": n_seeds_total,
        "n_splits": n_splits,
        "n_split_reports": len(test_means),
        "checkpoint": checkpoint,
        "report_path": ok_paths[0]
        if ok_paths
        else _report_relpath(_split_report_path(results_root, spec, 0)),
        "status": status,
    }
    return summary, split_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect GraphAny Wander-split results.")
    parser.add_argument("--output", type=str, default="")
    add_nofeat_flags(parser)
    args = parser.parse_args()

    folder = "nc-nofeat" if args.nofeat else "nc"
    results_root = WANDER_ROOT / "results" / "baselines" / "graphany" / folder
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
    partial = sum(1 for row in summaries if row["status"] == "partial")
    extra = f", {partial} partial" if partial else ""
    print(
        f"[{folder}] Wrote {output_path} and {splits_path} "
        f"({ok}/{len(summaries)} datasets complete{extra})"
    )


if __name__ == "__main__":
    main()
