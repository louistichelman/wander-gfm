"""Export Wander NC graphs as npy/npz bundles and AnyGraph node_data pickles.

Shared by GraphAny and AnyGraph NC harnesses. Splits come from
``DataSet.load(..., seed=42)`` (same as GraphPFN prepare). ``*-nofeat`` slugs
reuse the featured sibling ``split.npz`` and write empty features.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, is_undirected, to_undirected

_WANDER_ROOT = Path(__file__).resolve().parents[1]
if str(_WANDER_ROOT) not in sys.path:
    sys.path.insert(0, str(_WANDER_ROOT))

from baselines.paths import WANDER_ROOT  # noqa: E402
from baselines.registry import GraphPFNDatasetSpec, nc_eval_specs, resolve_nc_eval_spec  # noqa: E402

_WANDER_COO = "__wander_coo__"
_WANDER_NDARRAY = "__wander_ndarray__"


def dump_anygraph_pickle(path: Path, obj: Any) -> None:
    """Pickle scipy sparse / ndarrays without NumPy 2 ``numpy._core`` opcodes.

    AnyGraph's env is NumPy 1.x and cannot unpickle matrices written by NumPy 2.
    """
    from scipy.sparse import issparse

    if issparse(obj):
        mat = obj.tocoo()
        payload = {
            _WANDER_COO: 1,
            "shape": (int(mat.shape[0]), int(mat.shape[1])),
            "row": np.asarray(mat.row, dtype=np.int64).tobytes(),
            "col": np.asarray(mat.col, dtype=np.int64).tobytes(),
            "data": np.asarray(mat.data, dtype=np.float32).tobytes(),
        }
    else:
        arr = np.ascontiguousarray(obj, dtype=np.float32)
        payload = {
            _WANDER_NDARRAY: 1,
            "shape": tuple(int(x) for x in arr.shape),
            "dtype": "float32",
            "bytes": arr.tobytes(),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=4)


def load_anygraph_pickle(path: Path) -> Any:
    from scipy.sparse import coo_matrix

    with path.open("rb") as fh:
        obj = pickle.load(fh)
    if isinstance(obj, dict) and obj.get(_WANDER_COO):
        row = np.frombuffer(obj["row"], dtype=np.int64)
        col = np.frombuffer(obj["col"], dtype=np.int64)
        data = np.frombuffer(obj["data"], dtype=np.float32)
        return coo_matrix((data, (row, col)), shape=tuple(obj["shape"]))
    if isinstance(obj, dict) and obj.get(_WANDER_NDARRAY):
        return np.frombuffer(obj["bytes"], dtype=np.float32).reshape(obj["shape"]).copy()
    return obj


def add_nofeat_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--nofeat",
        "--no_features",
        dest="nofeat",
        action="store_true",
        default=False,
        help="Topology-only: select *-nofeat slugs / empty node features.",
    )


def labels_1d(y: np.ndarray | torch.Tensor) -> np.ndarray:
    """Integer class ids; argmax if one-hot; NaNs become -1."""
    arr = np.asarray(y)
    if arr.ndim == 2:
        finite = np.isfinite(arr).any(axis=-1)
        out = np.full(arr.shape[0], -1, dtype=np.int64)
        out[finite] = arr[finite].argmax(axis=-1).astype(np.int64)
        return out
    arr = arr.reshape(-1)
    out = np.full(arr.shape[0], -1, dtype=np.int64)
    if np.issubdtype(arr.dtype, np.floating):
        finite = np.isfinite(arr)
        out[finite] = arr[finite].astype(np.int64)
        return out
    out[:] = arr.astype(np.int64)
    return out


def num_classes_from_labels(labels: np.ndarray) -> int:
    valid = labels[labels >= 0]
    if valid.size == 0:
        raise ValueError("No finite class labels")
    return int(valid.max()) + 1


def undirected_edgelist(data: Data) -> np.ndarray:
    """Return undirected, coalesced edges as ``[E, 2]`` int64."""
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() else 0
    if edge_index.numel() == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if not is_undirected(edge_index, num_nodes=num_nodes):
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index = coalesce(edge_index, num_nodes=num_nodes)
    return edge_index.cpu().numpy().T.astype(np.int64)


def sibling_split_path(spec: GraphPFNDatasetSpec, out_dir: Path) -> Path | None:
    if not spec.ignore_features or not spec.slug.endswith("-nofeat"):
        return None
    base_slug = spec.slug[: -len("-nofeat")]
    sibling = out_dir.parent / base_slug / "split.npz"
    return sibling if sibling.is_file() else None


def split_masks_from_data(
    spec: GraphPFNDatasetSpec,
    data: Data,
    out_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sibling = sibling_split_path(spec, out_dir)
    if sibling is not None:
        split = np.load(sibling)
        return (
            split["train"].astype(bool),
            split["val"].astype(bool),
            split["test"].astype(bool),
        )
    for part in ("train", "val", "test"):
        if getattr(data, f"{part}_mask", None) is None:
            raise ValueError(f"{spec.slug}: missing {part}_mask")
    return (
        data.train_mask.cpu().numpy().astype(bool),
        data.val_mask.cpu().numpy().astype(bool),
        data.test_mask.cpu().numpy().astype(bool),
    )


def load_nc_pyg_data(spec: GraphPFNDatasetSpec, data_dir: Path, seed: int = 42) -> Data:
    """Load a transductive NC graph with paper splits (no aggressive PCA)."""
    from data.dataset import DataSet, get_datasetargs

    args = get_datasetargs(spec.registry_key)
    ds = DataSet(args, pca_target_dim=9999)
    bundle = ds.load(data_dir=str(data_dir), seed=seed)
    if bundle.train is None:
        raise RuntimeError(f"No transductive train graph for {spec.registry_key}")
    return bundle.train


def bundle_arrays(
    spec: GraphPFNDatasetSpec,
    data: Data,
    out_dir: Path,
) -> dict[str, np.ndarray]:
    n_nodes = int(data.num_nodes)
    labels = labels_1d(data.y.cpu().numpy())
    train, val, test = split_masks_from_data(spec, data, out_dir)
    if spec.ignore_features or getattr(data, "x", None) is None:
        features = np.zeros((n_nodes, 0), dtype=np.float32)
    else:
        features = data.x.cpu().numpy().astype(np.float32)
    return {
        "features": features,
        "targets": labels.astype(np.int64),
        "edgelist": undirected_edgelist(data),
        "train": train,
        "val": val,
        "test": test,
    }


def write_npy_bundle(out_dir: Path, arrays: dict[str, np.ndarray]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "features.npy", arrays["features"])
    np.save(out_dir / "targets.npy", arrays["targets"].astype(np.float32))
    np.save(out_dir / "edgelist.npy", arrays["edgelist"])
    np.savez(
        out_dir / "split.npz",
        train=arrays["train"],
        val=arrays["val"],
        test=arrays["test"],
    )


def write_anygraph_nc_pickles(
    out_dir: Path,
    arrays: dict[str, np.ndarray],
    *,
    nofeat: bool,
) -> None:
    """Write AnyGraph NC ``node_data`` pickles (class nodes appended)."""
    from scipy.sparse import coo_matrix

    labels = arrays["targets"].astype(np.int64)
    n_nodes = int(labels.shape[0])
    n_classes = num_classes_from_labels(labels)
    edgelist = arrays["edgelist"]
    train_mask = arrays["train"].astype(bool)
    test_mask = arrays["test"].astype(bool)

    n_aug = n_nodes + n_classes
    rows: list[int] = []
    cols: list[int] = []
    if edgelist.size:
        rows.extend(edgelist[:, 0].tolist())
        cols.extend(edgelist[:, 1].tolist())
    for node in np.flatnonzero(train_mask):
        lab = int(labels[int(node)])
        if lab < 0:
            continue
        class_id = n_nodes + lab
        rows.append(int(node))
        cols.append(class_id)
        rows.append(class_id)
        cols.append(int(node))
    data = np.ones(len(rows), dtype=np.float32)
    trn_mat = coo_matrix((data, (np.asarray(rows), np.asarray(cols))), shape=(n_aug, n_aug))

    tst_nodes = []
    tst_labs = []
    for node in np.flatnonzero(test_mask):
        lab = int(labels[int(node)])
        if lab < 0:
            continue
        tst_nodes.append(int(node))
        tst_labs.append(lab)
    tst_mat = coo_matrix(
        (
            np.ones(len(tst_nodes), dtype=np.float32),
            (np.asarray(tst_nodes, dtype=np.int64), np.asarray(tst_labs, dtype=np.int64)),
        ),
        shape=(n_nodes, n_classes),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    dump_anygraph_pickle(out_dir / "trn_mat.pkl", trn_mat)
    dump_anygraph_pickle(out_dir / "tst_mat.pkl", tst_mat)

    feats_path = out_dir / "feats.pkl"
    if nofeat:
        if feats_path.is_file():
            feats_path.unlink()
        return

    feat = arrays["features"]
    if feat.ndim != 2 or feat.shape[1] == 0:
        feat = np.ones((n_nodes, 1), dtype=np.float32)
    class_rows = np.zeros((n_classes, feat.shape[1]), dtype=np.float32)
    feats = np.concatenate([feat.astype(np.float32), class_rows], axis=0)
    dump_anygraph_pickle(feats_path, feats)


def export_nc_spec(
    spec: GraphPFNDatasetSpec,
    *,
    data_dir: Path,
    npy_root: Path | None = None,
    anygraph_root: Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    data = load_nc_pyg_data(spec, data_dir, seed=seed)
    npy_dir = (npy_root / spec.slug) if npy_root is not None else None
    # Prefer sibling split from the npy tree when exporting nofeat.
    split_anchor = npy_dir if npy_dir is not None else Path(".")
    arrays = bundle_arrays(spec, data, split_anchor)
    if npy_dir is not None:
        write_npy_bundle(npy_dir, arrays)
    if anygraph_root is not None:
        write_anygraph_nc_pickles(
            anygraph_root / "node_data" / spec.slug,
            arrays,
            nofeat=spec.ignore_features,
        )
    return {
        "slug": spec.slug,
        "n_nodes": int(arrays["targets"].shape[0]),
        "n_edges": int(arrays["edgelist"].shape[0]),
        "feat_dim": int(arrays["features"].shape[1]),
        "n_classes": num_classes_from_labels(arrays["targets"]),
    }


def _parse_specs(args: argparse.Namespace) -> list[GraphPFNDatasetSpec]:
    if args.dataset.strip():
        return [resolve_nc_eval_spec(args.dataset.strip(), nofeat=args.nofeat)]
    if args.datasets.strip():
        keys = [k.strip() for k in args.datasets.split(",") if k.strip()]
        return [resolve_nc_eval_spec(k, nofeat=args.nofeat) for k in keys]
    return list(nc_eval_specs(nofeat=True if args.nofeat else False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Wander NC graphs for GraphAny/AnyGraph.")
    parser.add_argument("--data_dir", type=str, default="raw_data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="", help="One slug or registry key.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated slugs or registry keys (default: all featured NC-eval specs).",
    )
    parser.add_argument(
        "--npy-root",
        type=str,
        default="",
        help="Write GraphAny npy bundles here (skip if empty).",
    )
    parser.add_argument(
        "--anygraph-root",
        type=str,
        default="",
        help="Write AnyGraph node_data pickles under this root (skip if empty).",
    )
    add_nofeat_flags(parser)
    args = parser.parse_args()

    specs = _parse_specs(args)
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (WANDER_ROOT / data_dir).resolve()
    npy_root = Path(args.npy_root) if args.npy_root else None
    anygraph_root = Path(args.anygraph_root) if args.anygraph_root else None
    if npy_root is not None and not npy_root.is_absolute():
        npy_root = (WANDER_ROOT / npy_root).resolve()
    if anygraph_root is not None and not anygraph_root.is_absolute():
        anygraph_root = (WANDER_ROOT / anygraph_root).resolve()

    print(f"Exporting {len(specs)} NC dataset(s)")
    for spec in specs:
        info = export_nc_spec(
            spec,
            data_dir=data_dir,
            npy_root=npy_root,
            anygraph_root=anygraph_root,
            seed=args.seed,
        )
        print(
            f"  {info['slug']}: n={info['n_nodes']} e={info['n_edges']} "
            f"C={info['n_classes']} F={info['feat_dim']}"
        )
    print("Done.")


if __name__ == "__main__":
    main()
