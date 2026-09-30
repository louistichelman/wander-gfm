#!/usr/bin/env python3
"""Prepare GraphPFN data layout: symlinks + Wander NC-protocol split.npz files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_WANDER_ROOT = Path(__file__).resolve().parents[2]
if str(_WANDER_ROOT) not in sys.path:
    sys.path.insert(0, str(_WANDER_ROOT))

from baselines.graphpfn.export import export_spec  # noqa: E402
from baselines.graphpfn.split_export import write_protocol_splits  # noqa: E402
from baselines.paths import WANDER_ROOT, GRAPHPFN_ROOT, require_checkout  # noqa: E402
from baselines.registry import GRAPHPFN_DATASET_SPECS, GraphPFNDatasetSpec, spec_by_slug  # noqa: E402


def _symlink_target(spec: GraphPFNDatasetSpec, data_dir: Path) -> Path:
    if spec.is_graphland:
        native = data_dir / "GraphLand" / spec.slug / "raw" / spec.slug
        if native.exists():
            return native
        return data_dir / "GraphLand" / spec.slug
    return data_dir / spec.data_name


def _ensure_symlink(link_path: Path, target: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink():
        if link_path.resolve() == target.resolve():
            return
        link_path.unlink()
    elif link_path.exists():
        raise FileExistsError(
            f"{link_path} exists and is not a symlink; remove it manually."
        )
    if not target.exists():
        raise FileNotFoundError(f"Missing dataset directory: {target}")
    link_path.symlink_to(target.resolve())


def prepare_dataset(
    spec: GraphPFNDatasetSpec,
    *,
    data_dir: Path,
    seed: int,
    skip_symlinks: bool,
    skip_splits: bool,
) -> None:
    out_root = GRAPHPFN_ROOT / "data"

    if spec.export_homogeneous:
        if not skip_splits:
            export_spec(spec, data_dir=data_dir, out_root=out_root, seed=seed)
        return

    link_path = out_root / spec.slug
    target = _symlink_target(spec, data_dir)

    if not skip_symlinks:
        _ensure_symlink(link_path, target)
        print(f"  symlink {link_path.name} -> {target}")

    if spec.split_kind == "external_npz" and not skip_splits:
        write_protocol_splits(
            spec, data_dir=data_dir, out_dir=out_root / spec.slug, seed=seed
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GraphPFN data + aligned splits.")
    parser.add_argument("--data_dir", type=str, default="raw_data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated Wander registry keys (default: all GraphPFN ICL datasets).",
    )
    parser.add_argument(
        "--slugs",
        type=str,
        default="",
        help="Comma-separated GraphPFN slugs (overrides --datasets).",
    )
    parser.add_argument("--skip_symlinks", action="store_true")
    parser.add_argument("--skip_splits", action="store_true")
    args = parser.parse_args()

    require_checkout("graphpfn")
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (WANDER_ROOT / data_dir).resolve()
    else:
        data_dir = data_dir.resolve()
    if args.slugs.strip():
        slugs = [s.strip() for s in args.slugs.split(",") if s.strip()]
        specs = [spec_by_slug(s) for s in slugs]
    elif args.datasets.strip():
        keys = {k.strip().upper() for k in args.datasets.split(",") if k.strip()}
        specs = [s for s in GRAPHPFN_DATASET_SPECS if s.registry_key in keys]
        missing = keys - {s.registry_key for s in specs}
        if missing:
            raise SystemExit(f"Unknown registry keys: {sorted(missing)}")
    else:
        specs = list(GRAPHPFN_DATASET_SPECS)

    print(f"Preparing {len(specs)} datasets (data_dir={data_dir}, seed={args.seed})")
    for spec in specs:
        print(spec.registry_key)
        prepare_dataset(
            spec,
            data_dir=data_dir,
            seed=args.seed,
            skip_symlinks=args.skip_symlinks,
            skip_splits=args.skip_splits,
        )
    print("Done.")


if __name__ == "__main__":
    main()
