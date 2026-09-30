"""In-process NodePFN inference on a raw npy bundle (released checkpoint).

The bundle stores ``load_base_dataset`` tensors. Preprocessing matches upstream
``nodepfn/node_classification.py``: optional PyG row-L1 (Actor / Coauthor
loaders), undirected edges, GCN-norm self-loops, ``SimpleConv``
smoothing, optional TSVD, then ``NodePFNClassifier``. Splits are Wander
protocol ``split.npz`` / ``split_{i}.npz``, not NodePFN's own loaders.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from baselines.nodepfn.configs import (
    CHECKPOINT_EPOCH,
    NOFEAT_STATIC_DIM,
    NODEPFN_MAX_CLASSES,
    NodePFNRunSpec,
    ensemble_size,
    spec_for_nc_slug,
    svd_algorithm,
    uses_row_l1_normalize,
)
from baselines.paths import NODEPFN_REPO_ROOT, NODEPFN_ROOT, require_checkout
from data.nc_splits import n_eval_nc_splits, split_npz_filename


def dump_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


def _ensure_nodepfn_on_path() -> None:
    require_checkout("nodepfn")
    root = str(NODEPFN_ROOT)
    # Wander/scripts is a path entry and would shadow NodePFN's ``scripts`` package.
    stale = sys.modules.get("scripts")
    origin = getattr(stale, "__file__", None) or ""
    if stale is not None and "nodepfn" not in origin.replace("\\", "/"):
        for key in list(sys.modules):
            if key == "scripts" or key.startswith("scripts."):
                del sys.modules[key]
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)


def _fix_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _labels_1d(y: np.ndarray) -> np.ndarray:
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


def _auroc(y: np.ndarray, probs: np.ndarray) -> float | None:
    try:
        if probs.shape[1] == 2:
            return float(roc_auc_score(y, probs[:, 1]))
        return float(roc_auc_score(y, probs, multi_class="ovr"))
    except ValueError:
        return None


def _mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None, None
    arr = np.asarray(nums, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def load_bundle(
    bundle_dir: Path,
    *,
    registry_key: str,
    split_index: int = 0,
) -> dict[str, np.ndarray | str | int]:
    features = np.load(bundle_dir / "features.npy")
    targets = _labels_1d(np.load(bundle_dir / "targets.npy"))
    edgelist = np.load(bundle_dir / "edgelist.npy")
    n_protocol = n_eval_nc_splits(registry_key)
    split_name = split_npz_filename(registry_key, split_index, n_protocol)
    split_path = bundle_dir / split_name
    if not split_path.is_file():
        raise FileNotFoundError(
            f"Missing {split_path}. Re-run: python baselines/nodepfn/prepare.py "
            f"--dataset {registry_key}"
        )
    split = np.load(split_path)
    return {
        "features": np.asarray(features),
        "targets": targets,
        "edgelist": np.asarray(edgelist),
        "train": split["train"].astype(bool),
        "val": split["val"].astype(bool),
        "test": split["test"].astype(bool),
        "split_file": split_name,
        "split_index": int(split_index),
    }


def _features_or_static(features: np.ndarray, *, n_nodes: int) -> np.ndarray:
    """Empty features become one shared all-ones column (structure-only)."""
    feat = np.asarray(features)
    if feat.ndim != 2:
        feat = feat.reshape(n_nodes, -1)
    if feat.shape[0] != n_nodes:
        raise ValueError(f"features has {feat.shape[0]} rows, expected {n_nodes}")
    if feat.shape[1] == 0:
        return np.ones((n_nodes, NOFEAT_STATIC_DIM), dtype=np.float32)
    return feat.astype(np.float32, copy=False)


def _maybe_row_l1_normalize(features: np.ndarray, slug: str) -> np.ndarray:
    """Match upstream ``T.NormalizeFeatures()`` on Actor / Coauthor."""
    if not uses_row_l1_normalize(slug) or features.shape[1] == 0:
        return features
    x = torch.as_tensor(features, dtype=torch.float32)
    denom = x.sum(dim=-1, keepdim=True).clamp(min=1.0)
    return (x / denom).cpu().numpy()


def _update_edge_index(
    edge_index: torch.Tensor, train_idx: torch.Tensor, query_idx: torch.Tensor
) -> torch.Tensor:
    """Match upstream ``update_edge_index``: train nodes then query nodes."""
    all_indices = torch.cat([train_idx, query_idx])
    old_to_new = torch.zeros(int(all_indices.max().item()) + 1, dtype=torch.long)
    old_to_new[train_idx] = torch.arange(len(train_idx))
    old_to_new[query_idx] = torch.arange(len(query_idx)) + len(train_idx)
    return old_to_new[edge_index]


def preprocess_like_nodepfn(
    features: np.ndarray,
    edgelist: np.ndarray,
    spec: NodePFNRunSpec,
    *,
    seed: int,
) -> tuple[np.ndarray, torch.Tensor]:
    """Undirected + GCN-norm + SimpleConv smoothing + optional TSVD."""
    from sklearn.decomposition import TruncatedSVD
    from torch_geometric.nn import SimpleConv
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    from torch_geometric.utils import add_self_loops, remove_self_loops, to_undirected

    n = int(features.shape[0])
    if edgelist.size:
        edge_index = torch.as_tensor(np.asarray(edgelist).T, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    edge_index = to_undirected(edge_index, num_nodes=n)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=n)
    edge_index, edge_weight = gcn_norm(
        edge_index,
        edge_weight=None,
        num_nodes=n,
        add_self_loops=True,
    )
    conv = SimpleConv(aggr="sum")
    feat = torch.as_tensor(features, dtype=torch.float32)
    for _ in range(int(spec.smoothing_steps)):
        feat = conv(feat, edge_index, edge_weight)
    original_features = int(feat.shape[1])
    x_np = feat.detach().cpu().numpy()
    if spec.dim_reduction != "none" and original_features > 0 and x_np.shape[0] > 1:
        n_components = min(int(spec.n_components), original_features, x_np.shape[0] - 1)
        if svd_algorithm(spec) == "arpack":
            n_components = min(n_components, original_features - 1, x_np.shape[0] - 1)
        if n_components >= 1:
            reducer = TruncatedSVD(
                n_components=n_components,
                algorithm=svd_algorithm(spec),
                random_state=seed,
            )
            x_np = reducer.fit_transform(x_np)
    return x_np, edge_index


def _mask_idx(mask: np.ndarray, labels: np.ndarray, *, labeled: bool) -> torch.Tensor:
    idx = np.flatnonzero(mask)
    if labeled:
        idx = idx[labels[idx] >= 0]
    return torch.from_numpy(idx.astype(np.int64))


def _score(
    y_true: np.ndarray, pred: np.ndarray, probs: np.ndarray
) -> dict[str, Any]:
    if y_true.size == 0:
        return {"accuracy": None, "roc-auc": None, "n": 0}
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "roc-auc": _auroc(y_true, probs),
        "n": int(y_true.size),
    }


def evaluate_slug(
    *,
    slug: str,
    bundle_dir: Path,
    registry_key: str,
    spec: NodePFNRunSpec | None = None,
    checkpoint_dir: Path | None = None,
    device: str | None = None,
    runs: int | None = None,
    cpu: bool = False,
    epoch: int = CHECKPOINT_EPOCH,
    batch_size_inference: int = 32,
    split_index: int = 0,
) -> dict[str, Any]:
    """Run NodePFN on one raw npy bundle with one Wander protocol split."""
    _ensure_nodepfn_on_path()
    from scripts.transformer_prediction_interface import NodePFNClassifier

    spec = spec if spec is not None else spec_for_nc_slug(slug)
    n_runs = int(spec.runs if runs is None else runs)
    n_ensemble = ensemble_size(spec)
    if checkpoint_dir is None:
        checkpoint_dir = NODEPFN_REPO_ROOT / "models_ckpts" / "nodepfn"
    if device is None:
        device = "cpu" if cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    bundle = load_bundle(
        bundle_dir, registry_key=registry_key, split_index=split_index
    )
    labels = bundle["targets"]
    n_nodes = int(labels.shape[0])
    n_classes = int(labels[labels >= 0].max()) + 1 if np.any(labels >= 0) else 0
    raw_features = np.asarray(bundle["features"])
    orig_feat_dim = 0 if raw_features.ndim != 2 else int(raw_features.shape[1])
    features = _features_or_static(raw_features, n_nodes=n_nodes)
    if orig_feat_dim > 0:
        features = _maybe_row_l1_normalize(features, slug)

    hparams = {
        "dim_reduction": spec.dim_reduction,
        "n_components": spec.n_components,
        "smoothing_steps": spec.smoothing_steps,
        "n_ensemble": n_ensemble,
        "svd_algorithm": svd_algorithm(spec),
        "runs": n_runs,
        "hparams_source": spec.hparams_source,
        "checkpoint_dir": str(checkpoint_dir),
        "epoch": epoch,
        "device": device,
        "row_l1_normalize": uses_row_l1_normalize(slug),
        "batch_size_inference": int(batch_size_inference),
        "split_index": int(split_index),
        "split_file": bundle["split_file"],
        "nofeat_fill": None
        if orig_feat_dim > 0
        else f"static_ones:{NOFEAT_STATIC_DIM}",
    }
    base = {
        "function": "baselines.nodepfn.evaluate.evaluate_slug",
        "slug": slug,
        "registry_key": registry_key,
        "split_index": int(split_index),
        "n_nodes": n_nodes,
        "n_classes": n_classes,
        "feat_dim": int(features.shape[1]),
        "hparams": hparams,
    }
    if n_classes > NODEPFN_MAX_CLASSES:
        return {
            **base,
            "status": "skipped",
            "reason": f"{n_classes} classes > NodePFN max ({NODEPFN_MAX_CLASSES})",
            "runs": [],
            "metrics": {"val": {}, "test": {}},
        }

    run_rows: list[dict[str, Any]] = []
    for run in range(n_runs):
        seed = run
        _fix_seed(seed)
        x, edge_index = preprocess_like_nodepfn(
            features, bundle["edgelist"], spec, seed=seed
        )
        train_idx = _mask_idx(bundle["train"], labels, labeled=True)
        valid_idx = _mask_idx(bundle["val"], labels, labeled=True)
        test_idx = _mask_idx(bundle["test"], labels, labeled=True)
        train_set = set(train_idx.tolist())
        query_idx = torch.tensor(
            [i for i in range(n_nodes) if i not in train_set], dtype=torch.long
        )
        if train_idx.numel() == 0 or query_idx.numel() == 0:
            raise RuntimeError(f"{slug}: empty train or query set")

        x_train = x[train_idx.numpy()]
        y_train = labels[train_idx.numpy()]
        x_query = x[query_idx.numpy()]
        y_query = labels[query_idx.numpy()]
        valid_mask = torch.isin(query_idx, valid_idx)
        test_mask = torch.isin(query_idx, test_idx)
        edge_index_run = _update_edge_index(edge_index, train_idx, query_idx)

        clf = NodePFNClassifier(
            device=device,
            base_path=str(checkpoint_dir),
            N_ensemble_configurations=n_ensemble,
            seed=seed,
            batch_size_inference=batch_size_inference,
            subsample_features=True,
            i=0,
            e=epoch,
        )
        t0 = time.time()
        clf.fit(x_train, y_train, edge_index_run, overwrite_warning=True)
        fit_time = time.time() - t0
        predictions = clf.predict(x_query, normalize_with_test=True)
        probabilities = clf.predict_proba(x_query, normalize_with_test=True)

        val_m = valid_mask.numpy()
        test_m = test_mask.numpy()
        val_metrics = _score(y_query[val_m], predictions[val_m], probabilities[val_m])
        test_metrics = _score(y_query[test_m], predictions[test_m], probabilities[test_m])
        run_rows.append(
            {
                "seed": seed,
                "fit_time": fit_time,
                "n_train": int(train_idx.numel()),
                "n_query": int(query_idx.numel()),
                "val": val_metrics,
                "test": test_metrics,
            }
        )
        print(
            f"  split={split_index} run {run + 1}/{n_runs}: "
            f"val acc={val_metrics['accuracy']!s} test acc={test_metrics['accuracy']!s}"
        )

    def _agg(split: str, key: str) -> dict[str, Any]:
        mean, std = _mean_std([row[split].get(key) for row in run_rows])
        n = run_rows[0][split].get("n") if run_rows else 0
        return {key: mean, f"{key}_std": std, "n": n}

    val_acc = _agg("val", "accuracy")
    test_acc = _agg("test", "accuracy")
    val_auc = _agg("val", "roc-auc")
    test_auc = _agg("test", "roc-auc")
    return {
        **base,
        "status": "ok",
        "runs": run_rows,
        "metrics": {
            "val": {**val_acc, "roc-auc": val_auc["roc-auc"], "roc-auc_std": val_auc["roc-auc_std"]},
            "test": {
                **test_acc,
                "roc-auc": test_auc["roc-auc"],
                "roc-auc_std": test_auc["roc-auc_std"],
            },
        },
    }
