"""Write Wander NC-protocol split.npz files for GraphPFN data dirs."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from baselines.registry import GraphPFNDatasetSpec
from data.nc_splits import (
    eval_nc_split_indices,
    n_eval_nc_splits,
    split_npz_filename,
)


def sibling_featured_dir(spec: GraphPFNDatasetSpec, out_dir: Path) -> Path | None:
    if not spec.ignore_features or not spec.slug.endswith("-nofeat"):
        return None
    return out_dir.parent / spec.slug[: -len("-nofeat")]


def copy_sibling_splits(src_dir: Path, dst_dir: Path) -> list[Path]:
    """Copy ``split.npz`` and ``split_*.npz`` from a featured sibling slug."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for src in sorted(src_dir.glob("split*.npz")):
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def _masks_from_bundle(data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parts = []
    for part in ("train", "val", "test"):
        mask = getattr(data, f"{part}_mask", None)
        if mask is None:
            raise RuntimeError(f"Missing {part}_mask")
        arr = mask.cpu().numpy().astype(bool)
        if arr.ndim != 1:
            raise RuntimeError(
                f"{part}_mask has ndim={arr.ndim}; DataSet should select a 1-D split"
            )
        parts.append(arr)
    return parts[0], parts[1], parts[2]


def load_protocol_masks(
    spec: GraphPFNDatasetSpec,
    *,
    data_dir: Path,
    seed: int,
    split_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from data.dataset import DataSet, get_datasetargs

    args = get_datasetargs(spec.registry_key)
    ds = DataSet(args, pca_target_dim=9999, nc_split_index=int(split_index))
    bundle = ds.load(data_dir=str(data_dir), seed=seed)
    if bundle.train is None:
        raise RuntimeError(f"No transductive train graph for {spec.registry_key}")
    return _masks_from_bundle(bundle.train)


def write_protocol_splits(
    spec: GraphPFNDatasetSpec,
    *,
    data_dir: Path,
    out_dir: Path,
    seed: int,
) -> list[Path]:
    """Write every Wander NC-protocol split under ``out_dir``.

    Multi-split datasets get ``split_{i}.npz`` plus a ``split.npz`` copy of
    split 0 (legacy GraphPFN configs). ``*-nofeat`` slugs reuse featured
    sibling files when present.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sibling = sibling_featured_dir(spec, out_dir)
    if sibling is not None:
        n = n_eval_nc_splits(spec.registry_key)
        expected = [split_npz_filename(spec.registry_key, i) for i in eval_nc_split_indices(spec.registry_key)]
        if n > 1:
            expected.append("split.npz")
        if all((sibling / name).is_file() for name in expected):
            copied = copy_sibling_splits(sibling, out_dir)
            print(f"  splits <- {sibling.name} ({len(copied)} files)")
            return copied

    indices = eval_nc_split_indices(spec.registry_key)
    n = n_eval_nc_splits(spec.registry_key)
    written: list[Path] = []
    for split_index in indices:
        train, val, test = load_protocol_masks(
            spec, data_dir=data_dir, seed=seed, split_index=split_index
        )
        split_path = out_dir / split_npz_filename(spec.registry_key, split_index, n)
        np.savez(split_path, train=train, val=val, test=test)
        written.append(split_path)
    if n > 1:
        alias = out_dir / "split.npz"
        shutil.copy2(out_dir / split_npz_filename(spec.registry_key, 0, n), alias)
        written.append(alias)
    print(
        f"  splits -> {out_dir} "
        f"({n} protocol split(s), files={len(written)})"
    )
    return written
