#!/usr/bin/env python3
"""Aggregate GraphPFN ICL evaluation reports into a summary table.

Multi-split datasets (official masks or GraphAny 0..4) are averaged over the
Wander NC protocol splits. Single-split datasets still average GraphPFN
ensemble seeds from ``<slug>/evaluation/report.json``.
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

from baselines.graphpfn.config_paths import (  # noqa: E402
    resolve_config_root,
    resolve_pca_dim,
    resolve_run_label,
    summary_csv_for_pca_dim,
)
from baselines.paths import WANDER_ROOT  # noqa: E402
from baselines.registry import GRAPHPFN_DATASET_SPECS  # noqa: E402
from data.nc_splits import (  # noqa: E402
    eval_nc_split_indices,
    n_eval_nc_splits,
    split_run_name,
)

REPO_ROOT = WANDER_ROOT

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
    "report_path",
    "status",
]


def _load_report(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _metric_key(spec) -> str:
    return "roc-auc" if spec.is_binary else "score"


def _seed_scores(report: dict, part: str, metric_key: str) -> list[float]:
    scores: list[float] = []
    for seed_report in report.get("reports", []):
        part_metrics = seed_report.get("metrics", {}).get(part, {})
        value = part_metrics.get(metric_key)
        if value is None and metric_key == "roc-auc":
            value = part_metrics.get("score")
        if value is not None:
            scores.append(float(value))
    return scores


def _summarize_scores(scores: list[float]) -> tuple[float | None, float | None]:
    if not scores:
        return None, None
    if len(scores) == 1:
        return scores[0], 0.0
    return statistics.mean(scores), statistics.pstdev(scores)


def _report_relpath(report_path: Path) -> str:
    try:
        return str(report_path.relative_to(REPO_ROOT))
    except ValueError:
        return str(report_path)


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
        "report_path": _report_relpath(report_path),
        "status": "missing",
    }


def _row_from_report(spec, report: dict, report_path: Path, metric_label: str) -> dict:
    metric_key = _metric_key(spec)
    metric = report.get("metric") or metric_label
    test_scores = _seed_scores(report, "test", metric_key)
    val_scores = _seed_scores(report, "val", metric_key)
    test_mean, test_std = _summarize_scores(test_scores)
    val_mean, val_std = _summarize_scores(val_scores)
    return {
        "registry_key": spec.registry_key,
        "slug": spec.slug,
        "metric": metric,
        "test_mean": "" if test_mean is None else f"{test_mean:.6f}",
        "test_std": "" if test_std is None else f"{test_std:.6f}",
        "val_mean": "" if val_mean is None else f"{val_mean:.6f}",
        "val_std": "" if val_std is None else f"{val_std:.6f}",
        "n_seeds": len(test_scores),
        "report_path": _report_relpath(report_path),
        "status": "ok" if test_scores else "empty",
        "_test_mean": test_mean,
        "_val_mean": val_mean,
    }


def _report_path_for_split(config_root: Path, spec, split_index: int) -> Path:
    sub = split_run_name(spec.registry_key, split_index, stem=spec.slug)
    return config_root / sub / "evaluation" / "report.json"


def collect_splits(
    config_root: Path,
    spec,
    *,
    legacy: bool = False,
) -> tuple[dict, list[dict]]:
    metric_label = "roc-auc" if spec.is_binary else "accuracy"
    n_splits = n_eval_nc_splits(spec.registry_key)
    indices = eval_nc_split_indices(spec.registry_key)
    if legacy or n_splits <= 1:
        report_path = config_root / spec.slug / "evaluation" / "report.json"
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
    for split_index in indices:
        report_path = _report_path_for_split(config_root, spec, split_index)
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
        "report_path": ok_paths[0] if ok_paths else _report_relpath(
            _report_path_for_split(config_root, spec, 0)
        ),
        "status": status,
    }
    return summary, split_rows


def collect_one(config_root: Path, spec, *, legacy: bool = False) -> dict:
    summary, _ = collect_splits(config_root, spec, legacy=legacy)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect GraphPFN ICL results.")
    parser.add_argument(
        "--config-root",
        type=str,
        default="",
        help="Config tree under graphpfn/ (default: from GRAPHPFN_PCA_DIM or GRAPHPFN_CONFIG_ROOT).",
    )
    parser.add_argument(
        "--pca-dim",
        type=str,
        default="",
        help="PCA variant: 64 (default), 0/none for no PCA. Ignored if --config-root is set.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output CSV path (default: results/baselines/graphpfn/<variant>/summary.csv).",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Read <slug>/evaluation/report.json only (old single-split 10-seed runs).",
    )
    args = parser.parse_args()

    config_root_arg = args.config_root or None
    pca_dim_arg = args.pca_dim if args.pca_dim else None
    config_root = resolve_config_root(pca_dim=pca_dim_arg, config_root=config_root_arg)
    pca_dim = resolve_pca_dim(pca_dim=pca_dim_arg, config_root=config_root_arg)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = summary_csv_for_pca_dim(pca_dim)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path = output_path.with_name("splits.csv")

    summaries: list[dict] = []
    split_rows: list[dict] = []
    for spec in GRAPHPFN_DATASET_SPECS:
        summary, splits = collect_splits(config_root, spec, legacy=args.legacy)
        summaries.append(summary)
        split_rows.extend(splits)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(summaries)

    with splits_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SPLIT_COLUMNS)
        writer.writeheader()
        writer.writerows(split_rows)

    ok = sum(1 for row in summaries if row["status"] == "ok")
    partial = sum(1 for row in summaries if row["status"] == "partial")
    label = resolve_run_label(pca_dim=pca_dim_arg, config_root=config_root_arg)
    extra = f", {partial} partial" if partial else ""
    print(
        f"[{label}] Wrote {output_path} and {splits_path} "
        f"({ok}/{len(summaries)} datasets complete{extra})"
    )


if __name__ == "__main__":
    main()
