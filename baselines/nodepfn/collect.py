#!/usr/bin/env python3
"""Aggregate NodePFN Wander-split reports into a summary CSV.

Multi-split datasets (official masks or GraphAny 0..4) are averaged over the
protocol splits. Single-split datasets still average TSVD/ensemble seeds in
``<slug>/report.json``.
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
from baselines.registry import nc_eval_specs, spec_by_slug  # noqa: E402
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
    "hparams_source",
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
    "hparams_source",
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


def _split_report_path(
    results_root: Path,
    spec,
    split_index: int,
    *,
    slug_root: Path | None = None,
) -> Path:
    root = slug_root if slug_root is not None else results_root / spec.slug
    if n_eval_nc_splits(spec.registry_key) <= 1:
        return root / "report.json"
    return root / f"{SPLIT_DIR_PREFIX}{int(split_index)}" / "report.json"


def _row_from_report(spec, report: dict, report_path: Path, metric_label: str) -> dict:
    metric_key = _metric_key(spec)
    test = report.get("metrics", {}).get("test", {})
    val = report.get("metrics", {}).get("val", {})
    if spec.is_binary and test.get("roc-auc") is None:
        metric_key = "accuracy"
        metric_label = "accuracy"
    test_v = test.get(metric_key)
    val_v = val.get(metric_key)
    n_seeds = len(report.get("runs") or [])
    status = report.get("status") or "ok"
    if status == "skipped":
        return {
            "registry_key": spec.registry_key,
            "slug": spec.slug,
            "metric": metric_label,
            "test_mean": "",
            "test_std": "",
            "val_mean": "",
            "val_std": "",
            "n_seeds": n_seeds,
            "hparams_source": str(report.get("hparams", {}).get("hparams_source", "")),
            "report_path": _report_relpath(report_path),
            "status": "skipped",
            "_test_mean": None,
            "_val_mean": None,
        }
    return {
        "registry_key": spec.registry_key,
        "slug": spec.slug,
        "metric": metric_label if test_v is None else metric_key,
        "test_mean": "" if test_v is None else f"{float(test_v):.6f}",
        "test_std": ""
        if test.get(f"{metric_key}_std") is None
        else f"{float(test[f'{metric_key}_std']):.6f}",
        "val_mean": "" if val_v is None else f"{float(val_v):.6f}",
        "val_std": ""
        if val.get(f"{metric_key}_std") is None
        else f"{float(val[f'{metric_key}_std']):.6f}",
        "n_seeds": n_seeds,
        "hparams_source": str(report.get("hparams", {}).get("hparams_source", "")),
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
        "hparams_source": "",
        "report_path": _report_relpath(report_path),
        "status": "missing",
    }


def collect_splits(
    results_root: Path, spec, *, slug_root: Path | None = None
) -> tuple[dict, list[dict]]:
    metric_label = _metric_key(spec)
    n_splits = n_eval_nc_splits(spec.registry_key)
    indices = eval_nc_split_indices(spec.registry_key)
    if n_splits <= 1:
        report_path = _split_report_path(
            results_root, spec, 0, slug_root=slug_root
        )
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
                "hparams_source": "",
                "report_path": _report_relpath(report_path),
                "status": "missing",
            }
            return summary, [_empty_split_row(spec, 0, report_path, metric_label)]
        row = _row_from_report(spec, report, report_path, metric_label)
        summary = {
            **{k: v for k, v in row.items() if not k.startswith("_")},
            "n_splits": n_splits,
            "n_split_reports": 1 if row["status"] in {"ok", "skipped"} else 0,
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
    hparams_source = ""
    skipped = False
    for split_index in indices:
        report_path = _split_report_path(
            results_root, spec, split_index, slug_root=slug_root
        )
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
        hparams_source = hparams_source or str(row.get("hparams_source", ""))
        if row["status"] == "skipped":
            skipped = True
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
    if skipped and not test_means:
        status = "skipped"
    elif not test_means:
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
        "hparams_source": hparams_source,
        "report_path": ok_paths[0]
        if ok_paths
        else _report_relpath(_split_report_path(results_root, spec, 0, slug_root=slug_root)),
        "status": status,
    }
    return summary, split_rows


GRID_COLUMNS = [
    "registry_key",
    "slug",
    "n_components",
    "smoothing_steps",
    "grid_tag",
    "metric",
    "test_mean",
    "test_std",
    "val_mean",
    "val_std",
    "n_seeds",
    "n_splits",
    "n_split_reports",
    "hparams_source",
    "report_path",
    "status",
]


def _parse_grid_tag(tag: str) -> tuple[int, int] | None:
    # tsvd15_smooth0
    if not tag.startswith("tsvd") or "_smooth" not in tag:
        return None
    left, right = tag.split("_smooth", 1)
    try:
        return int(left[len("tsvd") :]), int(right)
    except ValueError:
        return None


def collect_grid(results_root: Path | None = None, output_path: Path | None = None) -> Path:
    """Aggregate ``nc-grid/<slug>/tsvdK_smoothS`` reports."""
    grid_root = results_root or (WANDER_ROOT / "results" / "baselines" / "nodepfn" / "nc-grid")
    output_path = output_path or (grid_root / "summary.csv")
    rows: list[dict] = []
    for slug_dir in sorted(p for p in grid_root.iterdir() if p.is_dir()):
        slug = slug_dir.name
        try:
            spec = spec_by_slug(slug)
        except KeyError:
            continue
        for cell_dir in sorted(p for p in slug_dir.iterdir() if p.is_dir()):
            parsed = _parse_grid_tag(cell_dir.name)
            if parsed is None:
                continue
            n_components, smoothing_steps = parsed
            summary, _ = collect_splits(grid_root, spec, slug_root=cell_dir)
            rows.append(
                {
                    **{k: summary[k] for k in COLUMNS},
                    "n_components": n_components,
                    "smoothing_steps": smoothing_steps,
                    "grid_tag": cell_dir.name,
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=GRID_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    ok = sum(1 for row in rows if row["status"] == "ok")
    print(f"[nc-grid] Wrote {output_path} ({ok}/{len(rows)} cells complete)")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect NodePFN Wander-split results.")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Collect TSVD/smoothing grid reports under nc-grid/.",
    )
    add_nofeat_flags(parser)
    args = parser.parse_args()

    if args.grid:
        collect_grid(output_path=Path(args.output) if args.output else None)
        return

    folder = "nc-nofeat" if args.nofeat else "nc"
    results_root = WANDER_ROOT / "results" / "baselines" / "nodepfn" / folder
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
