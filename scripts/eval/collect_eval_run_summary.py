#!/usr/bin/env python3
"""Rebuild run-level eval summaries from per-dataset metrics.json files."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

_DONE_RE = re.compile(
    r"\[eval_only\]\s+(.+?)\s+done:\s+(.+)$",
    re.MULTILINE,
)
_METRIC_RE = re.compile(r"([A-Za-z0-9_@]+)=([0-9.]+)")


def _metric_summary(metrics: Dict[str, float]) -> str:
    skip = {"n_splits", "n_seeds"}
    parts = []
    for k, v in sorted(metrics.items()):
        if k.startswith("coverage/") or k in skip or k.endswith("_std"):
            continue
        std = metrics.get(f"{k}_std")
        if std is not None and not math.isnan(float(std)):
            parts.append(f"{k}={v:.4f} ± {float(std):.4f}")
        else:
            parts.append(f"{k}={v:.4f}")
    n_seeds = metrics.get("n_seeds")
    if n_seeds is not None and int(n_seeds) > 1:
        parts.append(f"n_seeds={int(n_seeds)}")
    n_splits = metrics.get("n_splits")
    if n_splits is not None and int(n_splits) > 1:
        parts.append(f"n_splits={int(n_splits)}")
    return ", ".join(parts)


def _parse_done_line(summary: str) -> Dict[str, float]:
    return {
        key: float(value)
        for key, value in _METRIC_RE.findall(summary)
    }


def _write_dataset_metrics(
    dataset_dir: Path,
    dataset: str,
    metrics: Dict[str, float],
    *,
    init_checkpoint: Optional[str],
    source: str,
) -> None:
    payload = {
        "experiment_name": dataset,
        "seed": 0,
        "init_checkpoint": init_checkpoint,
        "metrics": {dataset: metrics},
        "source": source,
    }
    metrics_path = dataset_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = _metric_summary(metrics)
    (dataset_dir / "metrics.txt").write_text(
        f"[eval_only] {dataset} done: {summary}\n",
        encoding="utf-8",
    )


_DEFAULT_LOG_GLOBS = (
    "Wander_eval_ev-*.out",
    "slurm_eval_ev-*.out",
    "slurm_eval_*.out",
)


def _parse_logs(logs_dir: Path, log_globs: Iterable[str]) -> Dict[str, Dict[str, float]]:
    parsed: Dict[str, Dict[str, float]] = {}
    seen_files: set[Path] = set()
    for pattern in log_globs:
        for log_path in sorted(logs_dir.glob(pattern)):
            resolved = log_path.resolve()
            if resolved in seen_files:
                continue
            seen_files.add(resolved)
            text = log_path.read_text(encoding="utf-8", errors="replace")
            for match in _DONE_RE.finditer(text):
                dataset = match.group(1).strip()
                metrics = _parse_done_line(match.group(2).strip())
                if metrics:
                    parsed[dataset] = metrics
    return parsed


def _read_init_checkpoint(run_dir: Path) -> Optional[str]:
    run_info = run_dir / "run_info.txt"
    if not run_info.is_file():
        return None
    for line in run_info.read_text(encoding="utf-8").splitlines():
        if line.startswith("CKPT="):
            return line.split("=", 1)[1].strip()
    return None


def _backfill_from_logs(
    run_dir: Path,
    logs_dir: Path,
    log_globs: Iterable[str],
) -> int:
    log_metrics = _parse_logs(logs_dir, log_globs)
    if not log_metrics:
        return 0

    init_checkpoint = _read_init_checkpoint(run_dir)
    written = 0
    for dataset_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        if (dataset_dir / "metrics.json").is_file():
            continue
        if any(p.is_dir() and p.name.startswith("split_") for p in dataset_dir.iterdir()):
            continue
        dataset = dataset_dir.name
        metrics = log_metrics.get(dataset)
        if metrics is None:
            continue
        _write_dataset_metrics(
            dataset_dir,
            dataset,
            metrics,
            init_checkpoint=init_checkpoint,
            source=f"backfill:{logs_dir.name}",
        )
        written += 1
    return written


def _read_eval_payload(dataset_dir: Path) -> Optional[dict]:
    """Prefer seed-averaged ``metrics_aggregate.json`` when present."""
    for name in ("metrics_aggregate.json", "metrics.json"):
        path = dataset_dir / name
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def _flatten_metric_dict(metric_dict: dict) -> Dict[str, float]:
    """Accept flat floats or ``{mean, std}`` from ``metrics_aggregate.json``."""
    out: Dict[str, float] = {}
    for key, value in (metric_dict or {}).items():
        name = str(key)
        if name.startswith("coverage/"):
            continue
        if isinstance(value, dict) and "mean" in value:
            out[name] = float(value["mean"])
            std = value.get("std")
            if std is not None:
                out[f"{name}_std"] = float(std)
        elif isinstance(value, (int, float)):
            out[name] = float(value)
    return out


def _payload_dataset_metrics(payload: dict, dataset: str) -> Dict[str, float]:
    metrics_by_name = payload.get("metrics", {}) or {}
    dataset_metrics = metrics_by_name.get(dataset)
    if dataset_metrics is None:
        dataset_metrics = next(
            (
                metric_dict
                for name, metric_dict in metrics_by_name.items()
                if name != "aggregate" and isinstance(metric_dict, dict)
            ),
            {},
        )
    flat = _flatten_metric_dict(dataset_metrics or {})
    seeds = payload.get("seeds")
    if (
        isinstance(seeds, list)
        and len(seeds) > 1
        and "n_seeds" not in flat
    ):
        flat["n_seeds"] = float(len(seeds))
    return flat


def _split_dir_index(path: Path) -> Optional[int]:
    name = path.name
    if not name.startswith("split_"):
        return None
    suffix = name[len("split_") :]
    return int(suffix) if suffix.isdigit() else None


def _mean_std(values: List[float]) -> Tuple[float, Optional[float]]:
    mean = float(statistics.fmean(values))
    if len(values) < 2:
        return mean, None
    return mean, float(statistics.stdev(values))


def _aggregate_split_metrics(dataset_dir: Path) -> Optional[dict]:
    """Mean ± std over ``split_*/metrics.json`` into ``dataset_dir/metrics.json``."""
    split_payloads: List[Tuple[int, dict, Dict[str, float]]] = []
    for child in sorted(dataset_dir.iterdir()):
        idx = _split_dir_index(child)
        if idx is None:
            continue
        payload = _read_eval_payload(child)
        if payload is None:
            continue
        metrics = _payload_dataset_metrics(payload, dataset_dir.name)
        if not metrics:
            continue
        split_payloads.append((idx, payload, metrics))
    if not split_payloads:
        return None

    keys = set(split_payloads[0][2])
    for _, _, metrics in split_payloads[1:]:
        keys &= set(metrics)
    keys = {k for k in keys if not k.endswith("_std") and k not in {"n_splits", "n_seeds"}}

    aggregated: Dict[str, float] = {"n_splits": float(len(split_payloads))}
    n_seeds_vals = [
        metrics["n_seeds"]
        for _, _, metrics in split_payloads
        if "n_seeds" in metrics
    ]
    if n_seeds_vals and all(v == n_seeds_vals[0] for v in n_seeds_vals):
        aggregated["n_seeds"] = float(n_seeds_vals[0])
    for key in sorted(keys):
        values = [metrics[key] for _, _, metrics in split_payloads]
        mean, std = _mean_std(values)
        aggregated[key] = mean
        if std is not None:
            aggregated[f"{key}_std"] = std

    init_checkpoint = next(
        (
            payload.get("init_checkpoint")
            for _, payload, _ in split_payloads
            if payload.get("init_checkpoint")
        ),
        None,
    )
    per_split = []
    for idx, payload, metrics in split_payloads:
        row = {"nc_split_index": idx, **metrics}
        if payload.get("nc_split_index") is not None:
            row["nc_split_index"] = int(payload["nc_split_index"])
        per_split.append(row)

    out = {
        "experiment_name": dataset_dir.name,
        "init_checkpoint": init_checkpoint,
        "n_splits": len(split_payloads),
        "metrics": {dataset_dir.name: aggregated},
        "per_split": per_split,
        "source": "aggregate_splits",
    }
    metrics_path = dataset_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = _metric_summary(aggregated)
    (dataset_dir / "metrics.txt").write_text(
        f"[eval_only] {dataset_dir.name} done: {summary}\n",
        encoding="utf-8",
    )
    return out


def _aggregate_run_splits(run_dir: Path) -> int:
    written = 0
    for dataset_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        if _aggregate_split_metrics(dataset_dir) is not None:
            written += 1
    return written


def _dataset_rows(run_dir: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for dataset_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        payload = _read_eval_payload(dataset_dir)
        if payload is None:
            continue
        dataset_metrics = _payload_dataset_metrics(payload, dataset_dir.name)
        if not dataset_metrics:
            continue
        rows.append((dataset_dir.name, _metric_summary(dataset_metrics)))
    return rows


def _write_summary(path: Path, rows: Iterable[Tuple[str, str]], *, readable: bool) -> None:
    lines = []
    for dataset, summary in rows:
        if readable:
            lines.append(f"[eval_only] {dataset} done: {summary}")
        else:
            lines.append(f"{dataset}\t{summary}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "eval_run_dir",
        type=Path,
        help="Run directory containing one subfolder per dataset.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=None,
        help="Optional SLURM log directory to backfill metrics.txt/json from .out files.",
    )
    parser.add_argument(
        "--log-glob",
        action="append",
        default=None,
        metavar="PATTERN",
        help=(
            "Glob pattern for SLURM stdout files under --logs-dir (repeatable). "
            f"Default: {', '.join(_DEFAULT_LOG_GLOBS)}"
        ),
    )
    args = parser.parse_args()

    run_dir = args.eval_run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Not a directory: {run_dir}")

    if args.logs_dir is not None:
        logs_dir = args.logs_dir.resolve()
        if not logs_dir.is_dir():
            raise SystemExit(f"Not a directory: {logs_dir}")
        log_globs = args.log_glob or list(_DEFAULT_LOG_GLOBS)
        backfilled = _backfill_from_logs(run_dir, logs_dir, log_globs)
        if backfilled:
            print(f"Backfilled metrics for {backfilled} dataset(s) from {logs_dir}")

    n_agg = _aggregate_run_splits(run_dir)
    if n_agg:
        print(f"Aggregated {n_agg} multi-split dataset(s)")

    rows = _dataset_rows(run_dir)
    _write_summary(run_dir / "eval_summary.tsv", rows, readable=False)
    _write_summary(run_dir / "eval_summary.txt", rows, readable=True)
    print(f"Wrote {len(rows)} dataset(s) to {run_dir / 'eval_summary.txt'}")


if __name__ == "__main__":
    main()
