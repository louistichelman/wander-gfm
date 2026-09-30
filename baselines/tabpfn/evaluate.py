"""In-process TabPFNv2 inference on a raw npy bundle (tabular; ignore graph).

Uses Wander protocol ``split.npz`` / ``split_{i}.npz``. Default preprocess is
StandardScaler + PCA(64) on all node features (edgelist unused).
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from baselines.nodepfn.evaluate import load_bundle  # noqa: E402
from baselines.tabpfn.configs import (  # noqa: E402
    DEFAULT_FIT_MODE,
    DEFAULT_MODEL_VERSION,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_PCA_DIM,
    DEFAULT_SEED,
    NOFEAT_STATIC_DIM,
    TABPFN_V2_MAX_CLASSES,
)


def dump_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


def _fix_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _auroc(y: np.ndarray, probs: np.ndarray) -> float | None:
    try:
        if probs.ndim == 1 or probs.shape[1] == 1:
            return float(roc_auc_score(y, probs.reshape(-1)))
        if probs.shape[1] == 2:
            return float(roc_auc_score(y, probs[:, 1]))
        return float(roc_auc_score(y, probs, multi_class="ovr"))
    except ValueError:
        return None


def _score(y_true: np.ndarray, pred: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    if y_true.size == 0:
        return {"accuracy": None, "roc-auc": None, "n": 0}
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "roc-auc": _auroc(y_true, probs),
        "n": int(y_true.size),
    }


def _features_or_static(features: np.ndarray, *, n_nodes: int) -> np.ndarray:
    feat = np.asarray(features)
    if feat.ndim != 2:
        feat = feat.reshape(n_nodes, -1)
    if feat.shape[0] != n_nodes:
        raise ValueError(f"features has {feat.shape[0]} rows, expected {n_nodes}")
    if feat.shape[1] == 0:
        return np.ones((n_nodes, NOFEAT_STATIC_DIM), dtype=np.float32)
    return feat.astype(np.float32, copy=False)


def reduce_features_pca(
    features: np.ndarray,
    *,
    pca_dim: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Column-standardize then PCA on **all** nodes (transductive dim reduction).

    If ``pca_dim <= 0`` or ``F <= pca_dim``, skip PCA and return standardized
    features when ``F > 1``, else the raw matrix.
    """
    x = np.asarray(features, dtype=np.float64)
    n, f = int(x.shape[0]), int(x.shape[1])
    meta: dict[str, Any] = {
        "pca_dim_requested": int(pca_dim),
        "feat_dim_in": f,
        "scaler": "none",
        "pca": "none",
        "feat_dim_out": f,
    }
    if f <= 0:
        return x.astype(np.float32), meta
    if f == 1:
        # Constant / single column: keep as-is (StandardScaler would zero it).
        return x.astype(np.float32), meta

    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    meta["scaler"] = "standard"
    if pca_dim <= 0 or f <= int(pca_dim):
        meta["feat_dim_out"] = f
        return xs.astype(np.float32), meta

    n_comp = min(int(pca_dim), f, max(1, n - 1))
    reducer = PCA(n_components=n_comp, random_state=seed)
    out = reducer.fit_transform(xs)
    meta["pca"] = "sklearn.PCA"
    meta["pca_dim_used"] = int(n_comp)
    meta["feat_dim_out"] = int(out.shape[1])
    return out.astype(np.float32), meta


def _model_version_enum(model_version: str):
    from tabpfn.constants import ModelVersion

    mapping = {
        "v2": ModelVersion.V2,
        "v2.5": ModelVersion.V2_5,
        "v2.6": ModelVersion.V2_6,
        "v3": ModelVersion.V3,
    }
    if model_version not in mapping:
        raise ValueError(f"Unknown model_version={model_version!r}")
    return mapping[model_version]


def _max_classes(model_version: str) -> int:
    if model_version == "v3":
        return 160
    return TABPFN_V2_MAX_CLASSES


def _mask_idx(mask: np.ndarray, labels: np.ndarray) -> np.ndarray:
    idx = np.flatnonzero(mask)
    return idx[labels[idx] >= 0]


def evaluate_slug(
    *,
    slug: str,
    bundle_dir: Path,
    registry_key: str,
    split_index: int = 0,
    pca_dim: int = DEFAULT_PCA_DIM,
    model_version: str = DEFAULT_MODEL_VERSION,
    seed: int = DEFAULT_SEED,
    device: str | None = None,
    cpu: bool = False,
    ignore_pretraining_limits: bool = True,
    fit_mode: str = DEFAULT_FIT_MODE,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
) -> dict[str, Any]:
    """Run TabPFN on one raw npy bundle with one Wander protocol split."""
    try:
        from tabpfn import TabPFNClassifier
    except ImportError as e:
        raise ImportError(
            "TabPFN is required for baselines/tabpfn. "
            "Install with: pip install -U tabpfn "
            f"(import error: {e})"
        ) from e

    if device is None:
        device = "cpu" if cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    bundle = load_bundle(
        bundle_dir, registry_key=registry_key, split_index=split_index
    )
    labels = np.asarray(bundle["targets"])
    n_nodes = int(labels.shape[0])
    n_classes = int(labels[labels >= 0].max()) + 1 if np.any(labels >= 0) else 0
    raw_features = np.asarray(bundle["features"])
    orig_feat_dim = 0 if raw_features.ndim != 2 else int(raw_features.shape[1])
    features = _features_or_static(raw_features, n_nodes=n_nodes)

    _fix_seed(seed)
    train_idx = _mask_idx(bundle["train"], labels)
    val_idx = _mask_idx(bundle["val"], labels)
    test_idx = _mask_idx(bundle["test"], labels)

    n_est_value: int | str
    if int(n_estimators) > 0:
        n_est_value = int(n_estimators)
    else:
        n_est_value = "auto"

    hparams = {
        "model_version": model_version,
        "pca_dim": int(pca_dim),
        "ignore_graph": True,
        "ignore_pretraining_limits": bool(ignore_pretraining_limits),
        "fit_mode": fit_mode,
        "n_estimators": n_est_value,
        "seed": int(seed),
        "device": device,
        "hparams_source": f"tabpfn-{model_version}-pca{pca_dim}",
        "split_index": int(split_index),
        "split_file": bundle["split_file"],
        "nofeat_fill": None
        if orig_feat_dim > 0
        else f"static_ones:{NOFEAT_STATIC_DIM}",
    }
    base = {
        "function": "baselines.tabpfn.evaluate.evaluate_slug",
        "slug": slug,
        "registry_key": registry_key,
        "split_index": int(split_index),
        "n_nodes": n_nodes,
        "n_classes": n_classes,
        "feat_dim": int(features.shape[1]),
        "hparams": hparams,
    }

    max_classes = _max_classes(model_version)
    if n_classes > max_classes:
        return {
            **base,
            "status": "skipped",
            "reason": f"{n_classes} classes > TabPFN {model_version} max ({max_classes})",
            "runs": [],
            "metrics": {"val": {}, "test": {}},
        }
    if train_idx.size == 0 or test_idx.size == 0:
        return {
            **base,
            "status": "skipped",
            "reason": "empty train or test",
            "runs": [],
            "metrics": {"val": {}, "test": {}},
        }

    print(
        f"  split={split_index}: PCA/features n={n_nodes} F_in={features.shape[1]} "
        f"pca_dim={pca_dim} device={device}",
        flush=True,
    )
    x, pca_meta = reduce_features_pca(features, pca_dim=pca_dim, seed=seed)
    hparams.update(pca_meta)
    base["feat_dim"] = int(x.shape[1])
    base["hparams"] = hparams

    X_train = x[train_idx]
    y_train = labels[train_idx].astype(np.int64, copy=False)
    X_val = x[val_idx] if val_idx.size else np.empty((0, x.shape[1]), dtype=np.float32)
    y_val = labels[val_idx].astype(np.int64, copy=False) if val_idx.size else np.empty(0, dtype=np.int64)
    X_test = x[test_idx]
    y_test = labels[test_idx].astype(np.int64, copy=False)

    clf = None
    fit_time = 0.0
    try:
        create_kwargs: dict[str, Any] = {
            "device": device,
            "random_state": seed,
            "ignore_pretraining_limits": ignore_pretraining_limits,
            "fit_mode": fit_mode,
        }
        if n_est_value != "auto":
            create_kwargs["n_estimators"] = n_est_value
        clf = TabPFNClassifier.create_default_for_version(
            _model_version_enum(model_version),
            **create_kwargs,
        )
        print(
            f"  split={split_index}: fitting TabPFN "
            f"(n_train={X_train.shape[0]}, F={X_train.shape[1]}, "
            f"n_estimators={n_est_value})",
            flush=True,
        )
        t0 = time.time()
        clf.fit(X_train, y_train)
        fit_time = time.time() - t0

        def _predict_split(X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
            if X.shape[0] == 0:
                return {"accuracy": None, "roc-auc": None, "n": 0}
            pred = clf.predict(X)
            probs = clf.predict_proba(X)
            return _score(y, pred, probs)

        print(f"  split={split_index}: predicting val/test", flush=True)
        val_metrics = _predict_split(X_val, y_val)
        test_metrics = _predict_split(X_test, y_test)
    except Exception as e:
        return {
            **base,
            "status": "error",
            "reason": str(e),
            "runs": [],
            "metrics": {"val": {}, "test": {}},
        }
    finally:
        if clf is not None:
            del clf
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    run_row = {
        "seed": int(seed),
        "fit_time": float(fit_time),
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
        "n_test": int(test_idx.size),
        "val": val_metrics,
        "test": test_metrics,
    }
    print(
        f"  split={split_index}: val acc={val_metrics['accuracy']!s} "
        f"test acc={test_metrics['accuracy']!s} fit_time={fit_time:.2f}s"
    )
    return {
        **base,
        "status": "ok",
        "runs": [run_row],
        "metrics": {
            "val": {
                "accuracy": val_metrics["accuracy"],
                "accuracy_std": 0.0 if val_metrics["accuracy"] is not None else None,
                "roc-auc": val_metrics["roc-auc"],
                "roc-auc_std": 0.0 if val_metrics["roc-auc"] is not None else None,
                "n": val_metrics["n"],
            },
            "test": {
                "accuracy": test_metrics["accuracy"],
                "accuracy_std": 0.0 if test_metrics["accuracy"] is not None else None,
                "roc-auc": test_metrics["roc-auc"],
                "roc-auc_std": 0.0 if test_metrics["roc-auc"] is not None else None,
                "n": test_metrics["n"],
            },
        },
    }
