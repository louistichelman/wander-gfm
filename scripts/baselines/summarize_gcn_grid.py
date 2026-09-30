"""Summarize GCN grid-search results across Wander protocol splits.

For each dataset, picks the hyperparameter config with the best validation
metric on every protocol split (same criterion as ``gcn_grid_summary.txt``),
then writes mean ± std. Single-split datasets keep one score (std 0).

Usage (from the repo root):
    python scripts/baselines/summarize_gcn_grid.py
    python scripts/baselines/summarize_gcn_grid.py --output results/baselines/gcn/grid.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parents[2]))
sys.path.insert(0, str(_SCRIPT_DIR))

from baselines.registry import GRAPHPFN_DATASET_SPECS
from data.nc_splits import SPLIT_DIR_PREFIX, eval_nc_split_indices, n_eval_nc_splits
from train_gcn_grid import _best_row, _eligible_dataset_keys, _ok_rows_from_csv

BINARY_REGISTRY_KEYS = frozenset()

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
    "lr",
    "dropout",
    "layer_norm",
    "skip_connections",
    "report_path",
    "status",
]


def _load_slug_metric(graphpfn_csv: Path) -> dict[str, tuple[str, str]]:
    mapping: dict[str, tuple[str, str]] = {}
    if not graphpfn_csv.exists():
        return mapping
    with open(graphpfn_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["registry_key"]] = (row["slug"], row["metric"])
    return mapping


def _slug_and_metric(
    key: str,
    slug_metric: dict[str, tuple[str, str]],
    *,
    ignore_features: bool,
) -> tuple[str, str]:
    for spec in GRAPHPFN_DATASET_SPECS:
        if spec.registry_key == key and spec.ignore_features is ignore_features:
            return spec.slug, ("roc-auc" if spec.is_binary else "accuracy")
    if key in slug_metric:
        slug, metric = slug_metric[key]
        if ignore_features and not slug.endswith("-nofeat"):
            slug = f"{slug}-nofeat"
        return slug, metric
    slug = key.lower().replace("_", "-")
    if ignore_features:
        slug = f"{slug}-nofeat"
    metric = "roc-auc" if key in BINARY_REGISTRY_KEYS else "accuracy"
    return slug, metric


def _dataset_dir(grid_root: Path, key: str, *, ignore_features: bool) -> Path:
    base = key.lower()
    if ignore_features:
        nofeat = grid_root / f"{base}_nofeat"
        if nofeat.is_dir():
            return nofeat
    return grid_root / base


def _split_csv_path(dataset_dir: Path, key: str, split_index: int) -> Path:
    if n_eval_nc_splits(key) <= 1:
        return dataset_dir / "gcn_grid.csv"
    return dataset_dir / f"{SPLIT_DIR_PREFIX}{int(split_index)}" / "gcn_grid.csv"


def _legacy_csv_path(dataset_dir: Path) -> Path:
    return dataset_dir / "gcn_grid.csv"


def _metric_fields(metric: str) -> tuple[str, str]:
    if metric == "roc-auc":
        return "test_auroc", "val_auroc"
    return "test_acc", "val_acc"


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def _summarize_scores(scores: list[float]) -> tuple[float | None, float | None]:
    if not scores:
        return None, None
    if len(scores) == 1:
        return scores[0], 0.0
    return statistics.mean(scores), statistics.pstdev(scores)


def _relpath(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def _best_scores(ok_rows: list[dict[str, str]], metric: str) -> tuple[dict[str, str], float | None, float | None]:
    best = _best_row(ok_rows)
    test_key, val_key = _metric_fields(metric)
    test_raw = best.get(test_key, "")
    val_raw = best.get(val_key, "")
    test_v = None if test_raw in ("", None) else float(test_raw)
    val_v = None if val_raw in ("", None) else float(val_raw)
    return best, test_v, val_v


def collect_dataset(
    *,
    key: str,
    grid_root: Path,
    slug_metric: dict[str, tuple[str, str]],
    repo_root: Path,
    ignore_features: bool,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    slug, metric = _slug_and_metric(key, slug_metric, ignore_features=ignore_features)
    dataset_dir = _dataset_dir(grid_root, key, ignore_features=ignore_features)
    n_splits = n_eval_nc_splits(key)
    indices = eval_nc_split_indices(key)
    summary_path = dataset_dir / "gcn_grid_summary.txt"

    split_rows: list[dict[str, str]] = []
    test_scores: list[float] = []
    val_scores: list[float] = []
    ok_paths: list[str] = []

    for split_index in indices:
        csv_path = _split_csv_path(dataset_dir, key, split_index)
        if not csv_path.is_file() and n_splits > 1 and split_index == 0:
            legacy = _legacy_csv_path(dataset_dir)
            if legacy.is_file():
                csv_path = legacy
        ok_rows = _ok_rows_from_csv(csv_path)
        rel = _relpath(csv_path, repo_root)
        if not ok_rows:
            split_rows.append(
                {
                    "registry_key": key,
                    "slug": slug,
                    "split_index": str(split_index),
                    "metric": metric,
                    "test_mean": "",
                    "test_std": "",
                    "val_mean": "",
                    "val_std": "",
                    "n_seeds": "0",
                    "lr": "",
                    "dropout": "",
                    "layer_norm": "",
                    "skip_connections": "",
                    "report_path": rel,
                    "status": "missing",
                }
            )
            continue
        best, test_v, val_v = _best_scores(ok_rows, metric)
        if test_v is not None:
            test_scores.append(test_v)
            ok_paths.append(rel)
        if val_v is not None:
            val_scores.append(val_v)
        split_rows.append(
            {
                "registry_key": key,
                "slug": slug,
                "split_index": str(split_index),
                "metric": metric,
                "test_mean": _fmt(test_v),
                "test_std": "0.000000" if test_v is not None else "",
                "val_mean": _fmt(val_v),
                "val_std": "0.000000" if val_v is not None else "",
                "n_seeds": "1" if test_v is not None else "0",
                "lr": str(best.get("lr", "")),
                "dropout": str(best.get("dropout", "")),
                "layer_norm": str(best.get("layer_norm", "")),
                "skip_connections": str(best.get("skip_connections", "")),
                "report_path": rel,
                "status": "ok" if test_v is not None else "empty",
            }
        )

    test_mean, test_std = _summarize_scores(test_scores)
    val_mean, val_std = _summarize_scores(val_scores)
    if not test_scores:
        status = "missing"
    elif len(test_scores) < n_splits:
        status = "partial"
    else:
        status = "ok"
    summary = {
        "registry_key": key,
        "slug": slug,
        "metric": metric,
        "test_mean": _fmt(test_mean),
        "test_std": _fmt(test_std),
        "val_mean": _fmt(val_mean),
        "val_std": _fmt(val_std),
        "n_seeds": str(len(test_scores)),
        "n_splits": str(n_splits),
        "n_split_reports": str(len(test_scores)),
        "report_path": ok_paths[0] if ok_paths else _relpath(summary_path, repo_root),
        "status": status,
    }
    return summary, split_rows


def summarize(
    *,
    grid_root: Path,
    slug_metric: dict[str, tuple[str, str]],
    repo_root: Path,
    ignore_features: bool,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    summaries: list[dict[str, str]] = []
    split_rows: list[dict[str, str]] = []
    for key in _eligible_dataset_keys():
        summary, splits = collect_dataset(
            key=key,
            grid_root=grid_root,
            slug_metric=slug_metric,
            repo_root=repo_root,
            ignore_features=ignore_features,
        )
        summaries.append(summary)
        split_rows.extend(splits)
    return summaries, split_rows


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Summarize GCN grid-search baselines over Wander protocol splits."
    )
    parser.add_argument(
        "--grid_root",
        type=Path,
        default=None,
        help="Directory containing per-dataset gcn_grid outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: results/baselines/gcn/grid.csv).",
    )
    parser.add_argument(
        "--graphpfn_csv",
        type=Path,
        default=repo_root / "results/baselines/graphpfn/nopca.csv",
        help="Reference CSV for slug/metric naming.",
    )
    parser.add_argument(
        "--nofeat",
        action="store_true",
        help="Summarize structure-only / nofeat grids.",
    )
    args = parser.parse_args()

    folder = "gcn_grid_nofeat" if args.nofeat else "gcn_grid"
    grid_root = args.grid_root or (repo_root / "results" / folder)
    if args.output is None:
        out_name = "grid_nofeat.csv" if args.nofeat else "grid.csv"
        split_name = "splits_nofeat.csv" if args.nofeat else "splits.csv"
        output_path = repo_root / "results" / "baselines" / "gcn" / out_name
        splits_path = output_path.with_name(split_name)
    else:
        output_path = args.output
        splits_path = output_path.with_name("splits.csv")

    summaries, split_rows = summarize(
        grid_root=grid_root,
        slug_metric=_load_slug_metric(args.graphpfn_csv),
        repo_root=repo_root,
        ignore_features=args.nofeat,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(summaries)
    with splits_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SPLIT_COLUMNS)
        writer.writeheader()
        writer.writerows(split_rows)

    n_ok = sum(r["status"] == "ok" for r in summaries)
    n_partial = sum(r["status"] == "partial" for r in summaries)
    extra = f", {n_partial} partial" if n_partial else ""
    print(
        f"Wrote {output_path} and {splits_path} "
        f"({n_ok}/{len(summaries)} datasets complete{extra})"
    )


if __name__ == "__main__":
    main()
