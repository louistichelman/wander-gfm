"""Grid-search GCN training on node-classification datasets.

Trains a 3-layer GCN (hidden dim 256) with Adam and early stopping on validation
accuracy (multiclass) or AUROC (binary). Sweeps learning rate, dropout, layer norm,
and skip connections.

Default protocol (``--hparam-mode first``): run the 60-config grid on split 0,
then retrain the best-val config on every other Wander protocol split.
Summarize averages those per-split test scores.

Usage (from the repo root):
    python scripts/baselines/train_gcn_grid.py --dataset CORA
    python scripts/baselines/train_gcn_grid.py --datasets CORA,TEXAS --output_dir results/gcn_grid
    python scripts/baselines/train_gcn_grid.py --dataset TEXAS --split 0
    python scripts/baselines/summarize_gcn_grid.py
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
import traceback
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.dataset import DataSet, apply_dummy_features, get_datasetargs, bundle_to_legacy_data
from data.datasets import DATASET_REGISTRY
from data.nc_splits import eval_nc_split_indices, n_eval_nc_splits, split_run_name

# Import metrics without loading experiment/__init__.py (avoids Wander/graph_walker).
import importlib.util as _importlib_util

_loss_metrics_path = Path(__file__).resolve().parents[2] / "experiment" / "loss_and_metrics.py"
_loss_spec = _importlib_util.spec_from_file_location("_loss_and_metrics", _loss_metrics_path)
assert _loss_spec and _loss_spec.loader
_loss_mod = _importlib_util.module_from_spec(_loss_spec)
_loss_spec.loader.exec_module(_loss_mod)
binary_roc_auc_score = _loss_mod.binary_roc_auc_score
macro_ovr_roc_auc_score = _loss_mod.macro_ovr_roc_auc_score

LEARNING_RATES = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
DROPOUTS = [0.0, 0.1, 0.2]
LAYER_NORM_OPTIONS = [False, True]
SKIP_OPTIONS = [False, True]

CSV_COLUMNS = [
    "dataset",
    "split_index",
    "lr",
    "dropout",
    "layer_norm",
    "skip_connections",
    "hidden_dim",
    "num_layers",
    "best_epoch",
    "epochs_ran",
    "train_acc",
    "val_acc",
    "test_acc",
    "train_auroc",
    "val_auroc",
    "test_auroc",
    "best_val_acc",
    "train_loss",
    "val_loss",
    "elapsed_sec",
    "seed",
    "ignore_features",
    "status",
    "error",
]


@dataclass(frozen=True)
class GridConfig:
    lr: float
    dropout: float
    layer_norm: bool
    skip_connections: bool

    def key(self) -> str:
        return (
            f"lr={self.lr:g}|dropout={self.dropout}|"
            f"layer_norm={int(self.layer_norm)}|skip={int(self.skip_connections)}"
        )


# From results/gcn_grid/old_best_settings.md (val-selected). Filtered graphs
# reuse the unfiltered row; CYCLES reuses MotifMixed.
OLD_BEST_HPARAMS: dict[str, GridConfig] = {
    "ACTOR": GridConfig(0.003, 0.2, True, True),
    "AMAZON_RATINGS": GridConfig(0.003, 0.2, False, True),
    "BRAZIL": GridConfig(0.001, 0.1, True, False),
    "CHAMELEON_FILTERED": GridConfig(0.003, 0.1, True, False),
    "CITESEER": GridConfig(0.003, 0.2, True, False),
    "COMPUTERS": GridConfig(0.001, 0.2, True, True),
    "CORA": GridConfig(0.003, 0.1, False, False),
    "CORNELL": GridConfig(0.001, 0.1, False, False),
    "CO_CS": GridConfig(0.003, 0.2, True, False),
    "CO_PHYSICS": GridConfig(0.001, 0.2, False, False),
    "EUROPE": GridConfig(3e-4, 0.1, False, False),
    "FULL_CORA": GridConfig(0.003, 0.2, True, False),
    "FULL_DBLP": GridConfig(0.003, 0.1, True, False),
    "HM_CATEGORIES": GridConfig(0.003, 0.2, True, False),
    "PHOTO": GridConfig(0.001, 0.2, True, False),
    "POKEC_REGIONS": GridConfig(0.003, 0.1, True, False),
    "POKEC_REGIONS_100K_TOP10": GridConfig(0.003, 0.1, True, False),
    "PUBMED": GridConfig(0.003, 0.0, False, False),
    "ROMAN_EMPIRE": GridConfig(0.003, 0.2, False, True),
    "SQUIRREL_FILTERED": GridConfig(3e-4, 0.1, True, False),
    "TEXAS": GridConfig(0.001, 0.1, False, False),
    "USA": GridConfig(0.001, 0.2, False, False),
    "WIKI_CS": GridConfig(0.001, 0.1, False, True),
    "WISCONSIN": GridConfig(0.001, 0.0, True, True),
    "MOTIF_MIXED": GridConfig(0.001, 0.0, True, False),
}

OLD_BEST_ALIASES: dict[str, str] = {
    "CYCLES": "MOTIF_MIXED",
    "MOTIF_MIXED": "MOTIF_MIXED",
}


def resolve_old_best_config(dataset_key: str) -> tuple[str, GridConfig]:
    """Return (source_key, config) from old_best_settings.md."""
    key = dataset_key.upper()
    source = OLD_BEST_ALIASES.get(key, key)
    if source.endswith("_FILTERED"):
        source = source[: -len("_FILTERED")]
    if source not in OLD_BEST_HPARAMS:
        raise KeyError(
            f"No old-best GCN hparams for {dataset_key} (looked up {source}). "
            "GRIDS is expected to run a fresh grid."
        )
    return source, OLD_BEST_HPARAMS[source]


def _eligible_dataset_keys() -> list[str]:
    keys: list[str] = []
    for name, module in DATASET_REGISTRY.items():
        if not getattr(module, "HAS_NODE_LABELS", False):
            continue
        if getattr(module, "HAS_NODE_FEATURES", False) or name in {"CYCLES", "GRIDS"}:
            keys.append(name)
    return sorted(keys)


def _dataset_stem(dataset_key: str, ignore_features: bool) -> str:
    return f"{dataset_key.lower()}{'_nofeat' if ignore_features else ''}"


def _split_out_dir(
    output_dir: Path, dataset_key: str, split_index: int, ignore_features: bool
) -> Path:
    return output_dir / split_run_name(
        dataset_key, split_index, stem=_dataset_stem(dataset_key, ignore_features)
    )


def _split_indices(dataset_key: str, *, split: Optional[int], mode: str) -> list[int]:
    n = n_eval_nc_splits(dataset_key)
    if split is not None:
        if split < 0 or split >= n:
            raise SystemExit(
                f"split={split} out of range for {dataset_key} ({n} protocol splits)"
            )
        return [split]
    return eval_nc_split_indices(dataset_key, mode=mode)


def _best_row(ok_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(ok_rows, key=lambda r: float(r["best_val_acc"]))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes"}:
        return True
    if text in {"0", "false", "f", "no"}:
        return False
    raise ValueError(f"Cannot parse boolean {value!r}")


def _config_from_row(row: dict[str, Any]) -> GridConfig:
    return GridConfig(
        lr=float(row["lr"]),
        dropout=float(row["dropout"]),
        layer_norm=_as_bool(row["layer_norm"]),
        skip_connections=_as_bool(row["skip_connections"]),
    )


def _load_best_config(
    output_dir: Path, dataset_key: str, split_index: int, ignore_features: bool
) -> GridConfig:
    csv_path = _split_out_dir(output_dir, dataset_key, split_index, ignore_features) / "gcn_grid.csv"
    ok_rows = _ok_rows_from_csv(csv_path)
    if not ok_rows:
        raise FileNotFoundError(
            f"No completed GCN grid at {csv_path}; run split {split_index} first."
        )
    return _config_from_row(_best_row(ok_rows))


def _split_has_results(
    output_dir: Path, dataset_key: str, split_index: int, ignore_features: bool
) -> bool:
    csv_path = _split_out_dir(output_dir, dataset_key, split_index, ignore_features) / "gcn_grid.csv"
    return bool(_ok_rows_from_csv(csv_path))


def get_labels_int(data: Data) -> Tensor:
    if data.y.ndim == 2:
        return data.y.argmax(dim=1)
    return data.y.long()


class GCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        dropout: float,
        layer_norm: bool,
        skip_connections: bool,
    ) -> None:
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels) if layer_norm else None
        self.dropout = dropout
        self.skip_connections = skip_connections
        self.skip_proj = (
            nn.Linear(in_channels, out_channels, bias=False)
            if skip_connections and in_channels != out_channels
            else None
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        h = self.conv(x, edge_index)
        if self.skip_connections:
            residual = self.skip_proj(x) if self.skip_proj is not None else x
            h = h + residual
        h = F.relu(h)
        if self.norm is not None:
            h = self.norm(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class GCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        *,
        dropout: float,
        layer_norm: bool,
        skip_connections: bool,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        blocks: list[nn.Module] = []
        blocks.append(
            GCNBlock(
                in_channels,
                hidden_channels,
                dropout=dropout,
                layer_norm=layer_norm,
                skip_connections=skip_connections,
            )
        )
        for _ in range(num_layers - 1):
            blocks.append(
                GCNBlock(
                    hidden_channels,
                    hidden_channels,
                    dropout=dropout,
                    layer_norm=layer_norm,
                    skip_connections=skip_connections,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        h = x
        for block in self.blocks:
            h = block(h, edge_index)
        return self.head(h)


@torch.no_grad()
def _accuracy(logits: Tensor, labels: Tensor, mask: Tensor) -> float:
    if int(mask.sum()) == 0:
        return float("nan")
    pred = logits[mask].argmax(dim=1)
    return float((pred == labels[mask]).float().mean().item())


@torch.no_grad()
def _auroc(logits: Tensor, labels: Tensor, mask: Tensor) -> float:
    if int(mask.sum()) == 0:
        return float("nan")
    masked_logits = logits[mask]
    masked_labels = labels[mask]
    if int(masked_labels.unique().numel()) < 2:
        return float("nan")
    prob_positive = F.softmax(masked_logits, dim=1)[:, 1].cpu().numpy()
    y_true = masked_labels.cpu().numpy()
    return binary_roc_auc_score(y_true, prob_positive)


@torch.no_grad()
def _macro_auroc(logits: Tensor, labels: Tensor, mask: Tensor) -> float:
    if int(mask.sum()) == 0:
        return float("nan")
    masked_logits = logits[mask]
    masked_labels = labels[mask]
    proba = F.softmax(masked_logits, dim=1).cpu().numpy()
    y_true = masked_labels.cpu().numpy()
    return macro_ovr_roc_auc_score(y_true, proba)


def _validation_metric(
    logits: Tensor, labels: Tensor, mask: Tensor, *, is_binary: bool
) -> float:
    if is_binary:
        return _auroc(logits, labels, mask)
    return _accuracy(logits, labels, mask)


@torch.no_grad()
def _loss(logits: Tensor, labels: Tensor, mask: Tensor) -> float:
    if int(mask.sum()) == 0:
        return float("nan")
    return float(F.cross_entropy(logits[mask], labels[mask]).item())


def train_one_config(
    data: Data,
    cfg: GridConfig,
    *,
    device: torch.device,
    hidden_dim: int,
    num_layers: int,
    max_epochs: int,
    patience: int,
    weight_decay: float,
    seed: int,
    is_binary: bool,
) -> dict[str, Any]:
    torch.manual_seed(seed)

    x = data.x
    edge_index = data.edge_index
    labels = get_labels_int(data)
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    if (
        x is None
        or x.numel() == 0
        or int(train_mask.sum()) == 0
        or int(val_mask.sum()) == 0
        or int(test_mask.sum()) == 0
    ):
        raise ValueError("Dataset missing features or non-empty train/val/test masks.")

    x = x.to(device)
    edge_index = edge_index.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    num_classes = int(labels.max().item()) + 1
    model = GCN(
        in_channels=int(x.shape[1]),
        hidden_channels=hidden_dim,
        out_channels=num_classes,
        num_layers=num_layers,
        dropout=cfg.dropout,
        layer_norm=cfg.layer_norm,
        skip_connections=cfg.skip_connections,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=weight_decay)

    best_state = copy.deepcopy(model.state_dict())
    best_val_metric = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    last_epoch = -1

    t0 = time.perf_counter()
    for epoch in range(max_epochs):
        last_epoch = epoch
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, edge_index)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(x, edge_index)
            val_metric = _validation_metric(logits, labels, val_mask, is_binary=is_binary)

        if val_metric > best_val_metric + 1e-12:
            best_val_metric = val_metric
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    elapsed = time.perf_counter() - t0
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)

    val_score = _validation_metric(logits, labels, val_mask, is_binary=is_binary)
    test_score = _validation_metric(logits, labels, test_mask, is_binary=is_binary)
    train_auroc = _auroc(logits, labels, train_mask) if is_binary else _macro_auroc(
        logits, labels, train_mask
    )
    val_auroc = _auroc(logits, labels, val_mask) if is_binary else _macro_auroc(
        logits, labels, val_mask
    )
    test_auroc = _auroc(logits, labels, test_mask) if is_binary else _macro_auroc(
        logits, labels, test_mask
    )

    return {
        "best_epoch": best_epoch,
        "epochs_ran": last_epoch + 1 if last_epoch >= 0 else 0,
        "train_acc": _accuracy(logits, labels, train_mask),
        "val_acc": val_score if not is_binary else _accuracy(logits, labels, val_mask),
        "test_acc": test_score if not is_binary else _accuracy(logits, labels, test_mask),
        "train_auroc": train_auroc,
        "val_auroc": val_auroc,
        "test_auroc": test_auroc,
        "best_val_acc": best_val_metric,
        "train_loss": _loss(logits, labels, train_mask),
        "val_loss": _loss(logits, labels, val_mask),
        "elapsed_sec": elapsed,
    }


def _load_completed_keys(jsonl_path: Path) -> set[str]:
    if not jsonl_path.exists():
        return set()
    done: set[str] = set()
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("status") == "ok":
                done.add(row["config_key"])
    return done


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


def _filter_grid(
    grid: list[GridConfig],
    *,
    lr: Optional[float],
    dropout: Optional[float],
    layer_norm: Optional[bool],
    skip_connections: Optional[bool],
) -> list[GridConfig]:
    filtered = grid
    if lr is not None:
        filtered = [c for c in filtered if c.lr == lr]
    if dropout is not None:
        filtered = [c for c in filtered if c.dropout == dropout]
    if layer_norm is not None:
        filtered = [c for c in filtered if c.layer_norm == layer_norm]
    if skip_connections is not None:
        filtered = [c for c in filtered if c.skip_connections == skip_connections]
    return filtered


def run_dataset_grid(
    *,
    dataset_key: str,
    data_dir: str,
    seed: int,
    pca_target_dim: int,
    hidden_dim: int,
    num_layers: int,
    max_epochs: int,
    patience: int,
    weight_decay: float,
    device: torch.device,
    output_dir: Path,
    resume: bool,
    split_index: int = 0,
    lr: Optional[float] = None,
    dropout: Optional[float] = None,
    layer_norm: Optional[bool] = None,
    skip_connections: Optional[bool] = None,
    fresh: bool = False,
    ignore_features: bool = False,
) -> list[dict[str, Any]]:
    dataset_args = get_datasetargs(dataset_key)
    label = dataset_args["name"]
    n_protocol = n_eval_nc_splits(dataset_key)
    out_subdir = _split_out_dir(output_dir, dataset_key, split_index, ignore_features)
    jsonl_path = out_subdir / "gcn_grid.jsonl"
    csv_path = out_subdir / "gcn_grid.csv"
    summary_path = out_subdir / "gcn_grid_summary.txt"

    native_nofeat = not bool(
        getattr(DATASET_REGISTRY.get(dataset_key), "HAS_NODE_FEATURES", False)
    )
    ds = DataSet(
        dataset_args,
        pca_target_dim=pca_target_dim,
        ignore_features=ignore_features,
        dummy_features=native_nofeat,
        nc_split_index=int(split_index),
    )
    data = bundle_to_legacy_data(ds.load(data_dir=data_dir, seed=seed))

    if data.y is None:
        raise ValueError(f"{label}: missing node labels.")
    if data.x is None or data.x.numel() == 0:
        if ignore_features or native_nofeat:
            data = apply_dummy_features(data)
        else:
            raise ValueError(f"{label}: missing node features.")
    if (
        data.train_mask is None
        or data.val_mask is None
        or data.test_mask is None
        or data.train_mask.shape[0] != data.num_nodes
    ):
        raise ValueError(f"{label}: missing node-level train/val/test masks.")

    labels = get_labels_int(data)
    is_binary = int(labels.max().item()) + 1 == 2
    val_metric_name = "AUROC" if is_binary else "accuracy"

    completed = _load_completed_keys(jsonl_path) if resume and not fresh else set()
    grid = [
        GridConfig(lr=cfg_lr, dropout=cfg_dropout, layer_norm=cfg_ln, skip_connections=cfg_skip)
        for cfg_lr, cfg_dropout, cfg_ln, cfg_skip in product(
            LEARNING_RATES, DROPOUTS, LAYER_NORM_OPTIONS, SKIP_OPTIONS
        )
    ]
    grid = _filter_grid(
        grid,
        lr=lr,
        dropout=dropout,
        layer_norm=layer_norm,
        skip_connections=skip_connections,
    )
    if not grid:
        raise ValueError("No grid configs match the provided hyperparameter filters.")

    rows: list[dict[str, Any]] = []
    if resume and not fresh and jsonl_path.exists():
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    print(
        f"{label} split {split_index}/{n_protocol}: "
        f"{len(grid)} configs ({len(completed)} already done)"
    )

    for i, cfg in enumerate(grid, start=1):
        config_key = cfg.key()
        if config_key in completed:
            continue

        row: dict[str, Any] = {
            "dataset": label,
            "split_index": int(split_index),
            "lr": cfg.lr,
            "dropout": cfg.dropout,
            "layer_norm": cfg.layer_norm,
            "skip_connections": cfg.skip_connections,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "seed": seed,
            "ignore_features": ignore_features,
            "config_key": config_key,
        }
        print(f"  [{i}/{len(grid)}] {config_key}")
        try:
            metrics = train_one_config(
                data,
                cfg,
                device=device,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                max_epochs=max_epochs,
                patience=patience,
                weight_decay=weight_decay,
                seed=seed,
                is_binary=is_binary,
            )
            row.update(metrics)
            row["status"] = "ok"
            row["error"] = ""
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
            traceback.print_exc()

        _write_jsonl(jsonl_path, row)
        rows.append(row)

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    _write_csv(rows, csv_path)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"GCN grid search: {label}\n")
        f.write(f"split_index={split_index}/{n_protocol}\n")
        f.write(f"ignore_features={ignore_features}\n")
        f.write(
            f"hidden_dim={hidden_dim}, num_layers={num_layers}, patience={patience}, "
            f"max_epochs={max_epochs}, weight_decay={weight_decay}, seed={seed}\n"
        )
        f.write(f"Completed configs: {len(ok_rows)}/{len(grid)}\n\n")
        if ok_rows:
            best = _best_row(ok_rows)
            f.write(f"Best by validation {val_metric_name}:\n")
            f.write(
                f"  lr={best['lr']}, dropout={best['dropout']}, "
                f"layer_norm={best['layer_norm']}, skip_connections={best['skip_connections']}\n"
            )
            f.write(
                f"  val={best['val_acc']:.4f}, test={best['test_acc']:.4f}, "
                f"val_auroc={best.get('val_auroc', float('nan')):.4f}, "
                f"test_auroc={best.get('test_auroc', float('nan')):.4f}, "
                f"train_acc={best['train_acc']:.4f}, epoch={best['best_epoch']}\n\n"
            )
        f.write(f"All configs (sorted by val {val_metric_name}):\n")
        for r in sorted(ok_rows, key=lambda x: float(x["best_val_acc"]), reverse=True):
            f.write(
                f"  lr={r['lr']:g} dropout={r['dropout']} "
                f"ln={int(bool(r['layer_norm']))} skip={int(bool(r['skip_connections']))}: "
                f"val={r['val_acc']:.4f} test={r['test_acc']:.4f} "
                f"val_auroc={r.get('val_auroc', float('nan')):.4f} "
                f"test_auroc={r.get('test_auroc', float('nan')):.4f} "
                f"train_acc={r['train_acc']:.4f} epoch={r['best_epoch']}\n"
            )

    print(f"  wrote {csv_path}")
    print(f"  wrote {summary_path}")
    return rows


def _ok_rows_from_csv(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok":
                rows.append(row)
    return rows


def write_dataset_split_average(
    *,
    output_dir: Path,
    dataset_key: str,
    ignore_features: bool,
    split_indices: list[int],
    hidden_dim: int,
    num_layers: int,
    patience: int,
    max_epochs: int,
    weight_decay: float,
    seed: int,
) -> None:
    """Write mean ± std of per-split best-val configs at the dataset root."""
    n_protocol = n_eval_nc_splits(dataset_key)
    if n_protocol <= 1:
        return
    dataset_args = get_datasetargs(dataset_key)
    label = dataset_args["name"]
    root = output_dir / _dataset_stem(dataset_key, ignore_features)
    summary_path = root / "gcn_grid_summary.txt"

    best_rows: list[dict[str, Any]] = []
    for split_index in split_indices:
        csv_path = _split_out_dir(output_dir, dataset_key, split_index, ignore_features) / "gcn_grid.csv"
        ok_rows = _ok_rows_from_csv(csv_path)
        if ok_rows:
            best_rows.append(_best_row(ok_rows))

    def _mean_std(key: str) -> tuple[float | None, float | None]:
        values = [float(r[key]) for r in best_rows if r.get(key) not in ("", None)]
        if not values:
            return None, None
        mean = sum(values) / len(values)
        if len(values) == 1:
            return mean, 0.0
        var = sum((v - mean) ** 2 for v in values) / len(values)
        return mean, var ** 0.5

    test_mean, test_std = _mean_std("test_acc")
    val_mean, val_std = _mean_std("val_acc")
    test_auroc_mean, test_auroc_std = _mean_std("test_auroc")
    val_auroc_mean, val_auroc_std = _mean_std("val_auroc")

    root.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"GCN grid search: {label} (Wander protocol splits)\n")
        f.write(f"ignore_features={ignore_features}\n")
        f.write(
            f"hidden_dim={hidden_dim}, num_layers={num_layers}, patience={patience}, "
            f"max_epochs={max_epochs}, weight_decay={weight_decay}, seed={seed}\n"
        )
        f.write(f"Protocol splits: {len(best_rows)}/{n_protocol} with a best-val config\n\n")
        if best_rows:
            f.write("Mean ± std of per-split best-val configs:\n")
            if test_mean is not None and test_std is not None:
                f.write(f"  test_acc={test_mean:.4f} ± {test_std:.4f}\n")
            if val_mean is not None and val_std is not None:
                f.write(f"  val_acc={val_mean:.4f} ± {val_std:.4f}\n")
            if test_auroc_mean is not None and test_auroc_std is not None:
                f.write(f"  test_auroc={test_auroc_mean:.4f} ± {test_auroc_std:.4f}\n")
            if val_auroc_mean is not None and val_auroc_std is not None:
                f.write(f"  val_auroc={val_auroc_mean:.4f} ± {val_auroc_std:.4f}\n")
            f.write("\nPer-split best:\n")
            for row in best_rows:
                f.write(
                    f"  split={row.get('split_index', '?')} "
                    f"lr={row['lr']} dropout={row['dropout']} "
                    f"ln={row['layer_norm']} skip={row['skip_connections']}: "
                    f"val={float(row['val_acc']):.4f} test={float(row['test_acc']):.4f}\n"
                )
    print(f"  wrote {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid-search GCN on node-classification graphs.")
    parser.add_argument("--dataset", type=str, default=None, help="Single DATASET_REGISTRY key.")
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated DATASET_REGISTRY keys (overrides --dataset).",
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca_target_dim", type=int, default=9999)
    parser.add_argument("--output_dir", type=str, default="results/gcn_grid")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (default: auto).")
    parser.add_argument("--resume", action="store_true", help="Skip configs already in JSONL.")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing JSONL completions (re-run filtered configs).",
    )
    parser.add_argument("--lr", type=float, default=None, help="Restrict grid to this learning rate.")
    parser.add_argument("--dropout", type=float, default=None, help="Restrict grid to this dropout.")
    parser.add_argument(
        "--layer_norm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Restrict grid to layer_norm on/off.",
    )
    parser.add_argument(
        "--skip_connections",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Restrict grid to skip_connections on/off.",
    )
    parser.add_argument(
        "--ignore_features",
        action="store_true",
        help="Drop node features and train structure-only GCN (constant dummy input).",
    )
    parser.add_argument(
        "--split",
        type=int,
        default=None,
        help="Run one Wander NC protocol split index (default: every protocol split).",
    )
    parser.add_argument(
        "--split-mode",
        choices=("all", "first"),
        default="all",
        help="all = official / GraphAny protocol splits; first = split 0 only.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print eligible registry keys and protocol split counts, then exit.",
    )
    parser.add_argument(
        "--hparam-mode",
        choices=("first", "per-split"),
        default="first",
        help=(
            "first = 60-config grid on split 0, then that best config on later splits "
            "(default). per-split = independent grid on every protocol split."
        ),
    )
    parser.add_argument(
        "--old-best",
        action="store_true",
        help=(
            "Skip the 60-config grid and use hparams from "
            "results/gcn_grid/old_best_settings.md (filtered graphs use the "
            "unfiltered row; CYCLES uses MotifMixed)."
        ),
    )
    args = parser.parse_args()

    if args.list:
        for key in _eligible_dataset_keys():
            print(f"{key}\tsplits={n_eval_nc_splits(key)}")
        return

    if args.dataset is None and args.datasets is None:
        raise SystemExit("Provide --dataset or --datasets (or --list).")

    if args.datasets:
        keys = [x.strip().upper() for x in args.datasets.split(",") if x.strip()]
    else:
        keys = [args.dataset.strip().upper()]

    eligible = set(_eligible_dataset_keys())
    unknown = [k for k in keys if k not in eligible]
    if unknown:
        raise SystemExit(
            f"Unknown or ineligible dataset keys: {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(eligible))}"
        )

    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    if args.ignore_features and args.output_dir == "results/gcn_grid":
        output_dir = Path("results/gcn_grid_nofeat")
    output_dir.mkdir(parents=True, exist_ok=True)

    for key in keys:
        indices = _split_indices(key, split=args.split, mode=args.split_mode)
        n_protocol = n_eval_nc_splits(key)
        preset_lr = args.lr
        preset_dropout = args.dropout
        preset_layer_norm = args.layer_norm
        preset_skip = args.skip_connections
        if args.old_best:
            source, cfg = resolve_old_best_config(key)
            if preset_lr is None:
                preset_lr = cfg.lr
            if preset_dropout is None:
                preset_dropout = cfg.dropout
            if preset_layer_norm is None:
                preset_layer_norm = cfg.layer_norm
            if preset_skip is None:
                preset_skip = cfg.skip_connections
            print(
                f"{key}: old-best hparams from {source}: {cfg.key()}"
            )
        print(f"=== {key} splits={indices}/{n_protocol} hparam_mode={args.hparam_mode} ===")
        try:
            for split_index in indices:
                print(f"--- {key} split {split_index}/{n_protocol} ---")
                if not args.fresh and _split_has_results(
                    output_dir, key, split_index, args.ignore_features
                ):
                    print(f"  skip split {split_index}: already has gcn_grid.csv")
                    continue
                lr = preset_lr
                dropout = preset_dropout
                layer_norm = preset_layer_norm
                skip_connections = preset_skip
                if (
                    args.hparam_mode == "first"
                    and split_index != 0
                    and lr is None
                    and dropout is None
                    and layer_norm is None
                    and skip_connections is None
                ):
                    reused = _load_best_config(
                        output_dir, key, 0, args.ignore_features
                    )
                    lr = reused.lr
                    dropout = reused.dropout
                    layer_norm = reused.layer_norm
                    skip_connections = reused.skip_connections
                    print(f"  reusing split 0 best config: {reused.key()}")
                elif lr is not None:
                    print(
                        f"  fixed config lr={lr:g} dropout={dropout} "
                        f"ln={int(bool(layer_norm))} skip={int(bool(skip_connections))}"
                    )
                run_dataset_grid(
                    dataset_key=key,
                    data_dir=args.data_dir,
                    seed=args.seed,
                    pca_target_dim=args.pca_target_dim,
                    hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    weight_decay=args.weight_decay,
                    device=device,
                    output_dir=output_dir,
                    resume=args.resume,
                    split_index=split_index,
                    lr=lr,
                    dropout=dropout,
                    layer_norm=layer_norm,
                    skip_connections=skip_connections,
                    fresh=args.fresh,
                    ignore_features=args.ignore_features,
                )
            write_dataset_split_average(
                output_dir=output_dir,
                dataset_key=key,
                ignore_features=args.ignore_features,
                split_indices=_split_indices(key, split=None, mode="all"),
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                patience=args.patience,
                max_epochs=args.max_epochs,
                weight_decay=args.weight_decay,
                seed=args.seed,
            )
        except Exception as exc:
            print(f"  dataset error: {exc}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
