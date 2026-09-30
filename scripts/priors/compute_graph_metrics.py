"""Synthetic prior complexity sweep and/or real-world graph metrics (structure, features, MLP).

Generates synthetic graphs with SCMPrior at multiple complexity levels (optional) and/or loads
all registered datasets, writing per-graph CSV rows for synthetics and a summary table with
real-world rows plus synthetic means (by task and complexity when sweeping).

Synthetic sweeps (unless ``--skip_synthetic``):
  - ``node_cls``: non-bipartite metrics (structure, features, labels, MLP, node splits).
  - ``lp_bipartite``: bipartite link-prediction metrics with ``bipartite_lp_prob=1``.

Real-world datasets use task-appropriate metrics automatically (node classification, KG LP,
anygraph bipartite, etc.).

Usage (from project root):
    python scripts/priors/compute_graph_metrics.py
    python scripts/priors/compute_graph_metrics.py --graphs_per_complexity 200
    python scripts/priors/compute_graph_metrics.py --output_dir results/graph_metrics
    python scripts/priors/compute_graph_metrics.py --fixed_complexity 0.5
    python scripts/priors/compute_graph_metrics.py --skip_real_world --synthetic_tasks node_cls
    python scripts/priors/compute_graph_metrics.py --no-skip_real_world --skip_synthetic --datasets CORA
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import multiprocessing as mp
import random
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, TextIO

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.dataset import DataSet, get_datasetargs
from data.datasets import DATASET_REGISTRY
from data.graph_bundle import is_link_prediction, resolve_split
from data.prior.prior_config_loader import PRIOR_CONFIG_NAMES

N_BFS_SAMPLES = 200
SLOW_METRIC_TIMEOUT_SEC: Optional[float] = 30 * 60

CSV_COLUMNS = [
    "complexity",
    "synthetic_task",
    "prior_source",
    "graph_idx",
    "num_nodes",
    "num_edges",
    "clustering_coeff",
    "approx_diameter",
    "num_features",
    "feature_variance",
    "feature_sparsity",
    "num_classes",
    "class_balance",
    "edge_homophily",
    "unbiased_homophily",
    "mlp_train_acc",
    "mlp_test_acc",
    "pct_train_nodes",
    "pct_val_nodes",
    "pct_test_nodes",
    "pct_train_edges",
    "pct_val_edges",
    "pct_test_edges",
    "num_relations",
    "relation_count_min",
    "relation_count_max",
    "relation_in_out_entity_ratio_avg",
    "lp_effective_candidate_pool_mean",
    "num_users",
    "num_items",
    "bipartite_density",
    "cold_start_exposure",
    "bipartite_feature_side_distance",
]

SUMMARY_COLUMNS = ["row_label"] + [c for c in CSV_COLUMNS if c != "graph_idx"]
SUMMARY_NUMERIC_KEYS = [
    c
    for c in CSV_COLUMNS
    if c not in ("graph_idx", "complexity", "synthetic_task", "prior_source")
]

SYNTHETIC_TASK_SPECS: dict[str, dict[str, Any]] = {
    "node_cls": {
        "prior_task_type": "node_cls",
        "bipartite_lp_prob": None,
        "summary_prefix": "synthetic_node_cls_mean",
    },
    "lp_bipartite": {
        "prior_task_type": "lp",
        "bipartite_lp_prob": 1.0,
        "summary_prefix": "synthetic_lp_bipartite_mean",
    },
}

@dataclass
class MetricProfile:
    is_knowledge_graph: bool
    is_link_prediction: bool
    is_bipartite: bool
    has_features: bool
    has_labels: bool
    num_forward_relations: int


class RunOutput:
    """Print to the real console; optionally duplicate to a summary file."""

    def __init__(self, log_path: Optional[Path]) -> None:
        self._log: Optional[TextIO] = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = open(log_path, "w", encoding="utf-8")

    def line(self, *args: object, sep: str = " ", end: str = "\n", console_only: bool = False) -> None:
        text = sep.join(str(a) for a in args) + end
        sys.__stdout__.write(text)
        sys.__stdout__.flush()
        if self._log is not None and not console_only:
            self._log.write(text)
            self._log.flush()

    def close(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None


def _timed_out(deadline: Optional[float]) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def _all_dataset_keys() -> list[str]:
    keys: list[str] = []
    for name, module in DATASET_REGISTRY.items():
        if (
            getattr(module, "load_base_dataset", None) is None
            and getattr(module, "load_graph_bundle", None) is None
        ):
            continue
        keys.append(name)
    return sorted(keys)


def _forward_edge_view(data: Data) -> tuple[torch.Tensor, Optional[torch.Tensor], int]:
    """Return forward ``edge_index``, optional forward ``edge_type``, and total edge count."""
    ei = data.edge_index
    et = getattr(data, "edge_type", None)
    n_total = int(ei.shape[1])
    if getattr(data, "has_inverse_edges", False):
        n_fwd = n_total // 2
        ei = ei[:, :n_fwd]
        if et is not None:
            et = et[:n_fwd]
    return ei, et, n_total


def _forward_mask(data: Data, mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    n_total = int(data.edge_index.shape[1])
    if len(mask) != n_total:
        return None
    if getattr(data, "has_inverse_edges", False):
        return mask[: n_total // 2]
    return mask


def _num_forward_relations(data: Data) -> int:
    nr = int(getattr(data, "num_relations", 1) or 1)
    if getattr(data, "has_inverse_edges", False):
        return max(1, nr // 2)
    return max(1, nr)


def _build_profile(data: Data, dataset_args: dict, registry_module: Any) -> MetricProfile:
    is_bipartite = bool(
        getattr(data, "anygraph_bipartite", False)
    )
    n_fwd_rel = _num_forward_relations(data)
    is_kg = bool(dataset_args.get("is_knowledge_graph", False))
    if is_bipartite:
        is_kg = True
    is_lp = is_link_prediction(data)
    has_x = data.x is not None and data.x.numel() > 0 and data.x.ndim == 2
    has_y = (
        data.y is not None
        and data.y.numel() > 0
        and data.train_mask is not None
        and data.test_mask is not None
        and len(data.train_mask) == int(data.num_nodes)
        and len(data.test_mask) == int(data.num_nodes)
    )
    return MetricProfile(
        is_knowledge_graph=is_kg,
        is_link_prediction=is_lp,
        is_bipartite=is_bipartite,
        has_features=has_x,
        has_labels=has_y and bool(dataset_args.get("has_node_labels", False)),
        num_forward_relations=n_fwd_rel,
    )


def get_labels_int(data: Data) -> torch.Tensor:
    if data.y.ndim == 2:
        return data.y.argmax(dim=1)
    return data.y.long()


def _undirected_structure_worker(
    n: int,
    edge_pairs: list[tuple[int, int]],
    n_bfs_samples: int,
    seed: int,
) -> dict[str, Any]:
    """CPU-only NetworkX metrics (runs in a subprocess for timeout enforcement)."""
    random.seed(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edge_pairs)

    clustering_coeff = float(nx.average_clustering(G)) if n > 0 else float("nan")

    sample_nodes = random.sample(range(n), k=min(n_bfs_samples, n)) if n > 0 else []
    max_length = 0
    for src in sample_nodes:
        lengths = nx.single_source_shortest_path_length(G, src)
        vals = list(lengths.values())
        if vals:
            max_length = max(max_length, max(vals))

    return {
        "clustering_coeff": clustering_coeff,
        "approx_diameter": int(max_length),
    }


def _run_subprocess_timed(
    fn: Callable[..., dict[str, Any]],
    args: tuple[Any, ...],
    *,
    timeout_sec: Optional[float],
    metric_name: str,
    out: Optional[RunOutput] = None,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(1)
    try:
        async_result = pool.apply_async(fn, args)
        if timeout_sec is None:
            return async_result.get()
        return async_result.get(timeout=timeout_sec)
    except mp.TimeoutError:
        if out is not None:
            out.line(f"  skipped {metric_name} (exceeded {timeout_sec / 60:.0f} min)")
        pool.terminate()
        return {
            "clustering_coeff": float("nan"),
            "approx_diameter": float("nan"),
        }
    finally:
        pool.close()
        pool.join()


def compute_graph_metrics(
    data: Data,
    *,
    seed: int,
    deadline: Optional[float] = None,
    out: Optional[RunOutput] = None,
) -> dict[str, Any]:
    n = int(data.num_nodes)
    ei_fwd, _, _ = _forward_edge_view(data)
    ei_undirected = to_undirected(ei_fwd, num_nodes=n)
    m_undirected = int(ei_undirected.shape[1] // 2)

    out_dict: dict[str, Any] = {
        "num_nodes": n,
        "num_edges": m_undirected,
        "clustering_coeff": float("nan"),
        "approx_diameter": float("nan"),
    }

    if n == 0 or m_undirected == 0:
        return out_dict

    if _timed_out(deadline):
        if out is not None:
            out.line("  skipped undirected structure metrics (deadline exceeded)")
        return out_dict

    one_way = ei_undirected[0] <= ei_undirected[1]
    edge_pairs = list(
        zip(
            ei_undirected[0, one_way].cpu().numpy().tolist(),
            ei_undirected[1, one_way].cpu().numpy().tolist(),
        )
    )

    if deadline is None and SLOW_METRIC_TIMEOUT_SEC is None:
        timeout: Optional[float] = None
    else:
        remaining = (
            (deadline - time.monotonic()) if deadline is not None else SLOW_METRIC_TIMEOUT_SEC
        )
        cap = SLOW_METRIC_TIMEOUT_SEC if SLOW_METRIC_TIMEOUT_SEC is not None else remaining
        timeout = max(1.0, min(cap, remaining))

    slow = _run_subprocess_timed(
        _undirected_structure_worker,
        (n, edge_pairs, N_BFS_SAMPLES, seed),
        timeout_sec=timeout,
        metric_name="undirected structure (clustering/diameter)",
        out=out,
    )
    out_dict.update(slow)
    return out_dict


def compute_feature_metrics(data: Data) -> dict[str, Any]:
    x = data.x
    if x is None or x.numel() == 0 or x.ndim != 2:
        return {
            "num_features": 0,
            "feature_variance": float("nan"),
            "feature_sparsity": float("nan"),
        }
    return {
        "num_features": int(x.shape[1]),
        "feature_variance": float(x.var().item()),
        "feature_sparsity": float((x == 0).float().mean().item()),
    }


def compute_label_metrics(data: Data) -> dict[str, Any]:
    labels = get_labels_int(data)
    n_classes = int(labels.max().item()) + 1

    counts = torch.bincount(labels, minlength=n_classes).float()
    probs = counts / counts.sum().clamp_min(1.0)
    if n_classes <= 1:
        class_balance = 1.0
    else:
        entropy = -(probs * torch.log2(probs.clamp_min(1e-12))).sum()
        class_balance = float((entropy / np.log2(n_classes)).item())

    n = int(data.num_nodes)
    ei = to_undirected(data.edge_index, num_nodes=n)
    one_way = ei[0] <= ei[1]
    src, dst = ei[0, one_way], ei[1, one_way]
    ls = labels[src]
    ld = labels[dst]
    n_e = int(src.numel())
    if n_e == 0:
        return {
            "num_classes": n_classes,
            "class_balance": class_balance,
            "edge_homophily": float("nan"),
            "unbiased_homophily": float("nan"),
        }

    edge_homophily = float((ls == ld).float().mean().item())

    c_i = np.zeros(n_classes, dtype=np.float64)
    for i in range(n_classes):
        c_i[i] = float(((ls == i) & (ld == i)).float().sum().item()) / n_e
    s_sum = float(np.sqrt(c_i).sum())
    sum_c = float(c_i.sum())
    num = s_sum * s_sum - 1.0
    den = s_sum * s_sum + 1.0 - 2.0 * sum_c
    if not math.isfinite(den) or abs(den) < 1e-15:
        unbiased_homophily = float("nan")
    else:
        unbiased_homophily = float(num / den)

    return {
        "num_classes": n_classes,
        "class_balance": class_balance,
        "edge_homophily": edge_homophily,
        "unbiased_homophily": unbiased_homophily,
    }


class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def _acc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == labels).float().mean().item())


def train_and_eval_mlp(
    data: Data,
    device: torch.device,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    *,
    deadline: Optional[float] = None,
    out: Optional[RunOutput] = None,
) -> dict[str, float]:
    nan_out = {"mlp_train_acc": float("nan"), "mlp_test_acc": float("nan")}
    x = data.x
    labels = get_labels_int(data)
    train_mask = data.train_mask
    test_mask = data.test_mask

    if (
        x is None
        or x.numel() == 0
        or train_mask is None
        or test_mask is None
        or train_mask.sum() == 0
        or test_mask.sum() == 0
    ):
        return nan_out

    if _timed_out(deadline):
        if out is not None:
            out.line("  skipped MLP training (deadline exceeded)")
        return nan_out

    x_d = x.to(device)
    y_d = labels.to(device)
    train_mask_d = train_mask.to(device)
    test_mask_d = test_mask.to(device)
    n_classes = int(labels.max().item()) + 1

    model = SimpleMLP(x.shape[1], hidden_dim, n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for _ in range(epochs):
        if _timed_out(deadline):
            if out is not None:
                out.line("  skipped MLP training (exceeded time limit)")
            return nan_out
        optimizer.zero_grad()
        logits = model(x_d)
        loss = F.cross_entropy(logits[train_mask_d], y_d[train_mask_d])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(x_d)
    return {
        "mlp_train_acc": _acc(logits[train_mask_d], y_d[train_mask_d]),
        "mlp_test_acc": _acc(logits[test_mask_d], y_d[test_mask_d]),
    }


def compute_node_split_pcts(data: Data) -> dict[str, float]:
    n = int(data.num_nodes)
    train_mask = data.train_mask
    val_mask = getattr(data, "val_mask", None)
    test_mask = data.test_mask
    if (
        train_mask is None
        or test_mask is None
        or len(train_mask) != n
        or len(test_mask) != n
    ):
        return {
            "pct_train_nodes": float("nan"),
            "pct_val_nodes": float("nan"),
            "pct_test_nodes": float("nan"),
        }
    pct_train = float(train_mask.float().mean().item())
    pct_test = float(test_mask.float().mean().item())
    pct_val = (
        float(val_mask.float().mean().item())
        if val_mask is not None and len(val_mask) == n
        else float("nan")
    )
    return {
        "pct_train_nodes": pct_train,
        "pct_val_nodes": pct_val,
        "pct_test_nodes": pct_test,
    }


def compute_edge_split_pcts(data: Data) -> dict[str, float]:
    n_total = int(data.edge_index.shape[1])
    train_mask = data.train_mask
    val_mask = getattr(data, "val_mask", None)
    test_mask = data.test_mask
    if train_mask is None or test_mask is None or len(train_mask) != n_total:
        return {
            "pct_train_edges": float("nan"),
            "pct_val_edges": float("nan"),
            "pct_test_edges": float("nan"),
        }

    train_f = _forward_mask(data, train_mask)
    val_f = _forward_mask(data, val_mask)
    test_f = _forward_mask(data, test_mask)
    if train_f is None or test_f is None:
        return {
            "pct_train_edges": float("nan"),
            "pct_val_edges": float("nan"),
            "pct_test_edges": float("nan"),
        }

    n_fwd = int(train_f.numel())
    if n_fwd == 0:
        return {
            "pct_train_edges": float("nan"),
            "pct_val_edges": float("nan"),
            "pct_test_edges": float("nan"),
        }

    return {
        "pct_train_edges": float(train_f.float().mean().item()),
        "pct_val_edges": (
            float(val_f.float().mean().item()) if val_f is not None else float("nan")
        ),
        "pct_test_edges": float(test_f.float().mean().item()),
    }


def compute_kg_metrics(data: Data, profile: MetricProfile) -> dict[str, Any]:
    nan_out = {
        "num_relations": float("nan"),
        "relation_count_min": float("nan"),
        "relation_count_max": float("nan"),
        "relation_in_out_entity_ratio_avg": float("nan"),
    }
    if not profile.is_knowledge_graph or profile.num_forward_relations <= 0:
        return nan_out

    ei, et, _ = _forward_edge_view(data)
    if et is None:
        et = torch.zeros(ei.shape[1], dtype=torch.long)

    n_rel = profile.num_forward_relations
    counts = torch.zeros(n_rel, dtype=torch.long)
    in_entities: list[set[int]] = [set() for _ in range(n_rel)]
    out_entities: list[set[int]] = [set() for _ in range(n_rel)]

    heads = ei[0].tolist()
    tails = ei[1].tolist()
    rels = et.tolist()
    for h, t, r in zip(heads, tails, rels):
        if r >= n_rel:
            continue
        counts[r] += 1
        out_entities[r].add(int(h))
        in_entities[r].add(int(t))

    if int(counts.sum()) == 0:
        return nan_out

    positive_counts = counts[counts > 0]
    ratios: list[float] = []
    for r in range(n_rel):
        n_out = len(out_entities[r])
        n_in = len(in_entities[r])
        if n_out > 0:
            ratios.append(n_in / n_out)

    return {
        "num_relations": int(n_rel),
        "relation_count_min": int(positive_counts.min().item()),
        "relation_count_max": int(positive_counts.max().item()),
        "relation_in_out_entity_ratio_avg": (
            float(np.mean(ratios)) if ratios else float("nan")
        ),
    }


def compute_directed_lp_metrics(data: Data, profile: MetricProfile) -> dict[str, float]:
    nan_out = {"lp_effective_candidate_pool_mean": float("nan")}
    if not profile.is_link_prediction or profile.is_bipartite:
        return nan_out
    if profile.is_knowledge_graph and profile.num_forward_relations > 1:
        return nan_out

    ei, _, _ = _forward_edge_view(data)
    n = int(data.num_nodes)
    n_fwd = int(ei.shape[1])

    visible = getattr(data, "visible_mask", None)
    if visible is not None and len(visible) == int(data.edge_index.shape[1]):
        vis_f = _forward_mask(data, visible)
    else:
        train_m = _forward_mask(data, data.train_mask)
        vis_f = train_m if train_m is not None else torch.zeros(n_fwd, dtype=torch.bool)

    if vis_f is None:
        return nan_out

    train_neigh: dict[int, set[int]] = defaultdict(set)
    heads = ei[0, vis_f].tolist()
    tails = ei[1, vis_f].tolist()
    for h, t in zip(heads, tails):
        if h != t:
            train_neigh[int(h)].add(int(t))

    test_m = _forward_mask(data, data.test_mask)
    if test_m is None or not bool(test_m.any()):
        return nan_out

    sources = ei[0, test_m].unique().tolist()
    pools: list[float] = []
    for s in sources:
        s = int(s)
        pool = float((n - 1) - len(train_neigh.get(s, set())))
        pools.append(max(0.0, pool))

    return {
        "lp_effective_candidate_pool_mean": float(np.mean(pools)) if pools else float("nan"),
    }


def compute_bipartite_metrics(data: Data, profile: MetricProfile) -> dict[str, Any]:
    nan_out = {
        "num_users": float("nan"),
        "num_items": float("nan"),
        "bipartite_density": float("nan"),
        "cold_start_exposure": float("nan"),
        "bipartite_feature_side_distance": float("nan"),
    }
    if not profile.is_bipartite:
        return nan_out

    offset = int(getattr(data, "anygraph_candidate_offset", 0))
    n = int(data.num_nodes)
    if offset <= 0 or offset >= n:
        return nan_out

    num_users = offset
    num_items = n - offset

    ei, _, _ = _forward_edge_view(data)
    train_m = _forward_mask(data, data.train_mask)
    test_m = _forward_mask(data, data.test_mask)
    if train_m is None or test_m is None:
        return {
            **nan_out,
            "num_users": float(num_users),
            "num_items": float(num_items),
        }

    n_train_edges = int(train_m.sum().item())
    density = n_train_edges / max(1, num_users * num_items)

    train_tails = set(int(t) for t in ei[1, train_m].tolist())
    test_tails = ei[1, test_m].tolist()
    if not test_tails:
        cold_frac = float("nan")
    else:
        cold = sum(1 for t in test_tails if int(t) not in train_tails)
        cold_frac = cold / len(test_tails)

    feat_dist = float("nan")
    x = data.x
    if x is not None and x.ndim == 2 and x.shape[0] == n:
        user_mean = x[:offset].mean(dim=0)
        item_mean = x[offset:].mean(dim=0)
        feat_dist = float(torch.dist(user_mean, item_mean).item())

    return {
        "num_users": float(num_users),
        "num_items": float(num_items),
        "bipartite_density": float(density),
        "cold_start_exposure": float(cold_frac),
        "bipartite_feature_side_distance": feat_dist,
    }


def _blank_metrics() -> dict[str, Any]:
    return {
        k: float("nan")
        for k in CSV_COLUMNS
        if k not in ("complexity", "graph_idx", "synthetic_task", "prior_source")
    }


def _prepare_lp_data_for_metrics(data: Data) -> Data:
    """Attach full edge_index and train/test masks for LP bipartite metrics."""
    test_e = getattr(data, "lp_test_edges", None)
    if test_e is None or test_e.numel() == 0:
        return data

    train_ei = data.edge_index
    n_train = int(train_ei.shape[1])
    n_test = int(test_e.shape[1])
    full_ei = torch.cat([train_ei, test_e], dim=1)

    train_mask = torch.zeros(n_train + n_test, dtype=torch.bool)
    train_mask[:n_train] = True
    test_mask = ~train_mask

    out = data.clone()
    out.edge_index = full_ei
    out.train_mask = train_mask
    out.test_mask = test_mask
    out.is_link_prediction = True
    return out


def _build_synthetic_profile(task_key: str, data: Data) -> MetricProfile:
    has_x = data.x is not None and data.x.numel() > 0 and data.x.ndim == 2
    if task_key == "node_cls":
        return MetricProfile(
            is_knowledge_graph=False,
            is_link_prediction=False,
            is_bipartite=False,
            has_features=has_x,
            has_labels=True,
            num_forward_relations=1,
        )
    if task_key == "lp_bipartite":
        return MetricProfile(
            is_knowledge_graph=False,
            is_link_prediction=True,
            is_bipartite=bool(getattr(data, "anygraph_bipartite", False)),
            has_features=has_x,
            has_labels=False,
            num_forward_relations=1,
        )
    raise ValueError(f"Unknown synthetic task key: {task_key!r}")


def compute_all_metrics(
    data: Data,
    profile: MetricProfile,
    *,
    train_device: torch.device,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    complexity: float | None,
    out: Optional[RunOutput] = None,
    skip_mlp: bool = False,
) -> dict[str, Any]:
    row: dict[str, Any] = _blank_metrics()
    if complexity is not None:
        row["complexity"] = complexity

    deadline = (
        time.monotonic() + SLOW_METRIC_TIMEOUT_SEC
        if SLOW_METRIC_TIMEOUT_SEC is not None
        else None
    )

    row.update(
        compute_graph_metrics(data, seed=seed, deadline=deadline, out=out)
    )

    if profile.has_features:
        row.update(compute_feature_metrics(data))

    if profile.has_labels:
        row.update(compute_label_metrics(data))
        if not skip_mlp:
            row.update(
                train_and_eval_mlp(
                    data,
                    train_device,
                    hidden_dim,
                    epochs,
                    lr,
                    weight_decay,
                    deadline=deadline,
                    out=out,
                )
            )

    if profile.has_labels:
        row.update(compute_node_split_pcts(data))

    if profile.is_link_prediction or (
        data.train_mask is not None
        and len(data.train_mask) == int(data.edge_index.shape[1])
    ):
        row.update(compute_edge_split_pcts(data))

    if profile.is_knowledge_graph:
        row.update(compute_kg_metrics(data, profile))

    row.update(compute_directed_lp_metrics(data, profile))
    row.update(compute_bipartite_metrics(data, profile))

    return row


def build_complexities(step: float) -> list[float]:
    n_steps = int(round(1.0 / step))
    vals = [round(i * step, 10) for i in range(n_steps + 1)]
    vals[0] = 0.0
    vals[-1] = 1.0
    return vals


def save_rows_csv(
    rows: list[dict],
    path: Path,
    *,
    extra_columns: tuple[str, ...] = (),
) -> None:
    fieldnames = list(extra_columns) + [c for c in CSV_COLUMNS if c not in extra_columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, None) for k in fieldnames})


def aggregate_synthetic_means(all_rows: list[dict]) -> list[dict]:
    by_key: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for r in all_rows:
        task = str(r.get("synthetic_task", "node_cls"))
        by_key[(task, float(r["complexity"]))].append(r)
    out: list[dict] = []
    for task, c in sorted(by_key.keys(), key=lambda x: (x[0], x[1])):
        group = by_key[(task, c)]
        spec = SYNTHETIC_TASK_SPECS.get(task, {})
        prefix = spec.get("summary_prefix", f"synthetic_{task}_mean")
        mean_row: dict[str, Any] = {
            "row_label": f"{prefix}_c={c:g}",
            "complexity": c,
            "synthetic_task": task,
        }
        for key in SUMMARY_NUMERIC_KEYS:
            vals = [row[key] for row in group if key in row]
            arr = np.array(vals, dtype=np.float64)
            mean_row[key] = float(np.nanmean(arr)) if vals else float("nan")
        out.append(mean_row)
    return out


def _fmt_cell(key: str, val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return "nan"
    if key in ("approx_diameter", "num_relations", "relation_count_min", "relation_count_max"):
        if isinstance(val, (int, float)) and math.isfinite(float(val)):
            return str(int(val))
    if key in ("num_users", "num_items", "num_nodes", "num_edges", "num_features", "num_classes"):
        if isinstance(val, (int, float)) and math.isfinite(float(val)):
            return str(int(val))
    if isinstance(val, float) and math.isfinite(val):
        return f"{val:.6f}"
    return str(val)


def write_summary_table(
    out: RunOutput,
    *,
    header_lines: list[str],
    rows: list[dict],
) -> None:
    for h in header_lines:
        out.line(h)
    out.line()
    if not rows:
        out.line("(no summary rows)")
        return

    col_widths = {c: len(c) for c in SUMMARY_COLUMNS}
    str_rows: list[list[str]] = []
    for row in rows:
        cells = []
        for c in SUMMARY_COLUMNS:
            cell = _fmt_cell(c, row.get(c))
            cells.append(cell)
            col_widths[c] = max(col_widths[c], len(cell))
        str_rows.append(cells)

    header = "  ".join(c.ljust(col_widths[c]) for c in SUMMARY_COLUMNS)
    sep = "-" * len(header)
    out.line(header)
    out.line(sep)
    for cells in str_rows:
        out.line("  ".join(c.ljust(col_widths[SUMMARY_COLUMNS[i]]) for i, c in enumerate(cells)))


def run_synthetic_sweep(
    *,
    task_key: str,
    complexities: list[float],
    graphs_per_complexity: int,
    output_dir: Path,
    pca_target_dim: int,
    row_wise_norming: bool,
    row_norm_mode: str = "l2",
    pca_before_normalization: bool,
    drop_constant_train_features: bool,
    prior_source: str,
    prior_device: torch.device,
    train_device: torch.device,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    get_default_fixed_hp: Callable[..., dict],
    get_default_sampled_hp: Callable[..., dict],
    SCMPrior: Any,
    prior_to_pyg_data: Callable[..., Data],
    prior_to_pyg_lp_data: Callable[..., Data],
    skip_mlp: bool = False,
) -> list[dict]:
    spec = SYNTHETIC_TASK_SPECS[task_key]
    prior_task_type = spec["prior_task_type"]
    bipartite_lp_prob = spec["bipartite_lp_prob"]

    out_dir = output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    per_complexity_dir = out_dir / "by_complexity"
    per_complexity_dir.mkdir(parents=True, exist_ok=True)

    task_rows: list[dict] = []

    for complexity in complexities:
        print(f"\n=== synthetic_task={task_key} complexity={complexity:.1f} ===")
        fixed_hp_kwargs: dict[str, Any] = {"complexity": complexity}
        if bipartite_lp_prob is not None:
            fixed_hp_kwargs["bipartite_lp_prob"] = bipartite_lp_prob
        fixed_hp = get_default_fixed_hp(**fixed_hp_kwargs)
        sampled_hp = get_default_sampled_hp(complexity=complexity)
        prior = SCMPrior(
            classification_only=True,
            fixed_hp=fixed_hp,
            sampled_hp=sampled_hp,
            device=str(prior_device),
        )

        level_rows: list[dict] = []
        n_errors = 0

        for graph_idx in range(graphs_per_complexity):
            graph = None
            X = None
            y = None
            d = None
            lp_meta = None
            try:
                if prior_task_type == "lp":
                    X, lp_meta = prior.get_batch(task_type="lp")
                    if lp_meta is None:
                        raise RuntimeError("Prior returned empty LP sample.")
                    if X is None and not lp_meta.get("drop_features"):
                        raise RuntimeError(
                            "Prior returned X=None without drop_features."
                        )
                    data = prior_to_pyg_lp_data(
                        X,
                        lp_meta,
                        pca_target_dim=pca_target_dim,
                        row_wise_norming=row_wise_norming,
                        row_norm_mode=row_norm_mode,
                        pca_before_normalization=pca_before_normalization,
                        drop_constant_train_features=drop_constant_train_features,
                    )
                    data = _prepare_lp_data_for_metrics(data)
                else:
                    graph, X, y, d, _, train_size = prior.get_batch(task_type="node_cls")
                    if graph is None:
                        raise RuntimeError("Prior returned graph=None.")
                    data = prior_to_pyg_data(
                        graph=graph,
                        X=X,
                        y=y,
                        d=int(d.item()),
                        train_size=int(train_size),
                        pca_target_dim=pca_target_dim,
                        row_wise_norming=row_wise_norming,
                        row_norm_mode=row_norm_mode,
                        pca_before_normalization=pca_before_normalization,
                        drop_constant_train_features=drop_constant_train_features,
                    )

                profile = _build_synthetic_profile(task_key, data)
                row = {
                    "complexity": complexity,
                    "synthetic_task": task_key,
                    "prior_source": prior_source,
                    "graph_idx": graph_idx,
                }
                row.update(
                    compute_all_metrics(
                        data,
                        profile,
                        train_device=train_device,
                        hidden_dim=hidden_dim,
                        epochs=epochs,
                        lr=lr,
                        weight_decay=weight_decay,
                        seed=seed,
                        complexity=None,
                        skip_mlp=skip_mlp,
                    )
                )
                level_rows.append(row)
                task_rows.append(row)

                if (graph_idx + 1) % 10 == 0:
                    print(
                        f"  processed {graph_idx + 1}/{graphs_per_complexity} "
                        f"(errors={n_errors})"
                    )
            except Exception:
                n_errors += 1
                traceback.print_exc()
            finally:
                if graph is not None:
                    del graph
                if X is not None:
                    del X
                if y is not None:
                    del y
                if d is not None:
                    del d
                if lp_meta is not None:
                    del lp_meta
                gc.collect()
                if train_device.type == "cuda":
                    torch.cuda.empty_cache()

        level_path = per_complexity_dir / f"complexity_{complexity:.1f}_{task_key}.csv"
        save_rows_csv(level_rows, level_path)

        print(
            f"Saved {len(level_rows)} rows for synthetic_task={task_key} "
            f"complexity={complexity:.1f} to {level_path} (errors={n_errors})"
        )

    return task_rows


def run_real_world_rows(
    *,
    keys: list[str],
    data_dir: str,
    seed: int,
    pca_target_dim: int,
    row_wise_norming: bool,
    row_norm_mode: str = "l2",
    train_device: torch.device,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    out: RunOutput,
    skip_mlp: bool = False,
) -> list[dict]:
    rows: list[dict] = []
    for key in keys:
        dataset_args = get_datasetargs(key)
        label = dataset_args["name"]
        registry_module = DATASET_REGISTRY[key]
        try:
            ds = DataSet(
                dataset_args,
                pca_target_dim=pca_target_dim,
                row_wise_norming=row_wise_norming,
                row_norm_mode=row_norm_mode,
            )
            bundle = ds.load(data_dir=data_dir, seed=seed)
            data = resolve_split(bundle, "train")
            profile = _build_profile(data, dataset_args, registry_module)
            if profile.has_labels:
                task_name = "node_cls"
            elif profile.is_bipartite:
                task_name = "lp_bipartite"
            else:
                task_name = "lp"

            row = compute_all_metrics(
                data,
                profile,
                train_device=train_device,
                hidden_dim=hidden_dim,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                seed=seed,
                complexity=float("nan"),
                out=out,
                skip_mlp=skip_mlp,
            )
            row["row_label"] = label
            row["prior_source"] = "real_world"
            row["synthetic_task"] = task_name
            row["graph_idx"] = len(rows)
            rows.append(row)
            out.line(f"{label}: ok")
        except Exception as e:
            out.line(f"{label}: error: {e}")
            traceback.print_exc()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic prior complexity sweep and/or real-world graph metrics"
    )
    parser.add_argument("--output_dir", type=str, default="results/graph_metrics")
    parser.add_argument("--graphs_per_complexity", type=int, default=300)
    parser.add_argument("--complexity_step", type=float, default=0.05)
    parser.add_argument(
        "--fixed_complexity",
        type=float,
        default=None,
        metavar="C",
        help=(
            "If set, only generate graphs at this complexity in [0, 1] "
            "(ignores --complexity_step)."
        ),
    )
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/data",
        help="Data root for PyG / OGB downloads (real-world branch).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for RNG and DataSet.load.")
    parser.add_argument(
        "--pca_target_dim",
        type=int,
        default=9999,
        help="PCA target dim (9999 ≈ no PCA for typical feature sizes).",
    )
    parser.add_argument(
        "--row_wise_norming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use row-wise L2 normalization after PCA (default: on, matching "
            "training with --row_wise_norming). Use --no-row_wise_norming for "
            "column-wise z-score on train nodes."
        ),
    )
    parser.add_argument(
        "--row_norm_mode",
        type=str,
        choices=["l1", "l2"],
        default="l2",
        help="Row normalization type when --row_wise_norming: l1 or l2 (default).",
    )
    parser.add_argument(
        "--pca_before_normalization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When True (default): smoothing -> PCA -> normalization. "
            "When False: smoothing -> normalization -> PCA."
        ),
    )
    parser.add_argument(
        "--drop_constant_train_features",
        action="store_true",
        default=False,
        help="Drop feature columns constant on training nodes (final pipeline step).",
    )
    parser.add_argument(
        "--prior_config",
        type=str,
        default="default",
        choices=list(PRIOR_CONFIG_NAMES),
        help=(
            "Prior config for synthetic generation. "
            "graphpfn/nodepfn use native GraphPFN/NodePFN backends for node_cls."
        ),
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated DATASET_REGISTRY keys (default: all registered datasets).",
    )
    parser.add_argument(
        "--skip_real_world",
        action="store_true",
        help="Skip real-world metrics (default: run them).",
    )
    parser.add_argument(
        "--skip_synthetic",
        action="store_true",
        help="Skip synthetic prior generation (no per-graph CSVs).",
    )
    parser.add_argument(
        "--synthetic_tasks",
        type=str,
        default="all",
        help=(
            "Comma-separated synthetic sweeps: node_cls, lp_bipartite, or all "
            "(default: all)."
        ),
    )
    parser.add_argument(
        "--prior_device",
        type=str,
        default="cpu",
        help=(
            "Device for SCMPrior graph/feature generation (default: cpu). "
            "Use cpu: GraphSampler asserts graph tensors match this device; "
            "some samplers still produce CPU graphs, so cuda here often fails "
            "with AssertionError in graph.py."
        ),
    )
    parser.add_argument(
        "--summary_file",
        type=str,
        default="results/graph_metrics_summary.txt",
        help="Path for the text summary table.",
    )
    parser.add_argument(
        "--no_summary_file",
        action="store_true",
        help="Do not write the summary text file (console only).",
    )
    parser.add_argument(
        "--metric_timeout_min",
        type=float,
        default=30.0,
        help=(
            "Max minutes per dataset for slow metrics (structure, MLP); skip if exceeded. "
            "Use 0 to disable the timeout."
        ),
    )
    parser.add_argument(
        "--skip_mlp",
        action="store_true",
        help="Skip the MLP probe (mlp_train_acc / mlp_test_acc stay NaN).",
    )
    args = parser.parse_args()

    global SLOW_METRIC_TIMEOUT_SEC
    if args.metric_timeout_min <= 0:
        SLOW_METRIC_TIMEOUT_SEC = None
    else:
        SLOW_METRIC_TIMEOUT_SEC = max(1.0, args.metric_timeout_min * 60.0)

    if args.skip_real_world and args.skip_synthetic:
        raise SystemExit("Nothing to do: both --skip_real_world and --skip_synthetic are set.")

    if args.fixed_complexity is not None:
        if args.fixed_complexity < 0 or args.fixed_complexity > 1:
            raise ValueError("--fixed_complexity must be in [0, 1].")
    elif not args.skip_synthetic:
        if args.complexity_step <= 0 or args.complexity_step > 1:
            raise ValueError("--complexity_step must be in (0, 1].")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior_device = torch.device(args.prior_device)

    if not args.skip_synthetic:
        try:
            from data.prior.dataset import SCMPrior
            from data.prior.prior_config_loader import load_prior_config
            from data.prior_to_pyg import prior_to_pyg_data, prior_to_pyg_lp_data

            prior_config_mod = load_prior_config(args.prior_config)
            get_default_fixed_hp = prior_config_mod.get_default_fixed_hp
            get_default_sampled_hp = prior_config_mod.get_default_sampled_hp
        except ModuleNotFoundError as e:
            raise SystemExit(
                "Synthetic metrics need prior dependencies (e.g. pip install xgboost): "
                f"{e}"
            ) from e

    summary_path: Optional[Path] = None if args.no_summary_file else Path(args.summary_file)
    summary_out = RunOutput(summary_path)
    try:
        real_rows: list[dict] = []
        if not args.skip_real_world:
            keys = _all_dataset_keys()
            if args.datasets:
                wanted = {x.strip().upper() for x in args.datasets.split(",") if x.strip()}
                unknown = wanted - set(DATASET_REGISTRY.keys())
                if unknown:
                    summary_out.line(
                        "Warning: unknown dataset keys (skipped): "
                        + ", ".join(sorted(unknown))
                    )
                keys = [k for k in keys if k in wanted]
                if not keys:
                    summary_out.line("No datasets left after --datasets filter.")
                    raise SystemExit(1)

            summary_out.line(f"Real-world datasets ({len(keys)}): loading from {args.data_dir!r}")
            real_rows = run_real_world_rows(
                keys=keys,
                data_dir=args.data_dir,
                seed=args.seed,
                pca_target_dim=args.pca_target_dim,
                row_wise_norming=args.row_wise_norming,
                row_norm_mode=args.row_norm_mode,
                train_device=train_device,
                hidden_dim=args.hidden_dim,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                out=summary_out,
                skip_mlp=args.skip_mlp,
            )
            real_csv_name = (
                "all_graph_metrics.csv" if args.skip_synthetic else "real_world_metrics.csv"
            )
            real_csv = Path(args.output_dir) / real_csv_name
            save_rows_csv(real_rows, real_csv, extra_columns=("row_label",))
            summary_out.line(f"Saved {len(real_rows)} real-world rows to {real_csv}")
            summary_out.line()

        all_synthetic_rows: list[dict] = []
        if not args.skip_synthetic:
            print(f"Prior (generation): {prior_device}")
            print(f"MLP training: {train_device}")

            if args.synthetic_tasks.strip().lower() == "all":
                synthetic_task_keys = list(SYNTHETIC_TASK_SPECS.keys())
            else:
                synthetic_task_keys = [
                    x.strip() for x in args.synthetic_tasks.split(",") if x.strip()
                ]
                unknown = set(synthetic_task_keys) - set(SYNTHETIC_TASK_SPECS.keys())
                if unknown:
                    raise SystemExit(
                        "Unknown --synthetic_tasks keys: "
                        + ", ".join(sorted(unknown))
                        + f" (expected: {', '.join(SYNTHETIC_TASK_SPECS.keys())}, all)"
                    )
                if not synthetic_task_keys:
                    raise SystemExit("No synthetic tasks left after --synthetic_tasks filter.")

            out_dir = Path(args.output_dir)
            if args.fixed_complexity is not None:
                complexities = [args.fixed_complexity]
            else:
                complexities = build_complexities(args.complexity_step)

            print(f"Synthetic tasks: {', '.join(synthetic_task_keys)}")
            print(f"prior_config={args.prior_config}")
            for task_key in synthetic_task_keys:
                task_rows = run_synthetic_sweep(
                    task_key=task_key,
                    complexities=complexities,
                    graphs_per_complexity=args.graphs_per_complexity,
                    output_dir=out_dir,
                    pca_target_dim=args.pca_target_dim,
                    row_wise_norming=args.row_wise_norming,
                    row_norm_mode=args.row_norm_mode,
                    pca_before_normalization=args.pca_before_normalization,
                    drop_constant_train_features=args.drop_constant_train_features,
                    prior_source=args.prior_config,
                    prior_device=prior_device,
                    train_device=train_device,
                    hidden_dim=args.hidden_dim,
                    epochs=args.epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    seed=args.seed,
                    get_default_fixed_hp=get_default_fixed_hp,
                    get_default_sampled_hp=get_default_sampled_hp,
                    SCMPrior=SCMPrior,
                    prior_to_pyg_data=prior_to_pyg_data,
                    prior_to_pyg_lp_data=prior_to_pyg_lp_data,
                    skip_mlp=args.skip_mlp,
                )
                all_synthetic_rows.extend(task_rows)

            all_rows_path = out_dir / "all_graph_metrics.csv"
            save_rows_csv(all_synthetic_rows, all_rows_path)
            print(f"\nDone. Full table: {all_rows_path}")

        synthetic_summary = aggregate_synthetic_means(all_synthetic_rows) if all_synthetic_rows else []
        summary_rows = real_rows + synthetic_summary

        mode_bits = []
        if not args.skip_real_world:
            mode_bits.append("real-world")
        if not args.skip_synthetic:
            mode_bits.append("synthetic")
        sweep_desc = (
            f"fixed_c={args.fixed_complexity}"
            if args.fixed_complexity is not None
            else f"step={args.complexity_step}"
        )
        dev_line = (
            f"prior_device={prior_device}, train_device={train_device}, "
            f"pca_target_dim={args.pca_target_dim}, row_wise_norming={args.row_wise_norming}, "
            f"pca_before_normalization={args.pca_before_normalization}, "
            f"prior_config={args.prior_config}, seed={args.seed}, "
            f"metric_timeout_min={args.metric_timeout_min}, skip_mlp={args.skip_mlp}"
            if not args.skip_synthetic
            else (
                f"train_device={train_device}, pca_target_dim={args.pca_target_dim}, "
                f"row_wise_norming={args.row_wise_norming}, seed={args.seed}, "
                f"metric_timeout_min={args.metric_timeout_min}, skip_mlp={args.skip_mlp}"
            )
        )
        header_lines = [
            f"Graph metrics summary: {', '.join(mode_bits) or 'none'}",
            dev_line,
        ]
        if not args.skip_synthetic:
            task_desc = (
                "all"
                if args.synthetic_tasks.strip().lower() == "all"
                else args.synthetic_tasks
            )
            header_lines.append(
                f"Synthetic: tasks={task_desc}, graphs_per_complexity="
                f"{args.graphs_per_complexity}, {sweep_desc}, "
                f"hidden_dim={args.hidden_dim}, epochs={args.epochs}"
            )
        write_summary_table(summary_out, header_lines=header_lines, rows=summary_rows)
    finally:
        summary_out.close()


if __name__ == "__main__":
    main()
