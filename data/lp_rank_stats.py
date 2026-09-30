"""Accumulate filtered-ranking MRR / Hits@K, including per-relation breakdowns."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Mapping, Optional, Sequence

import torch
from torch import Tensor

#: Metric names emitted by :meth:`RankAccum.as_metrics` for a *single relation*
#: (e.g. ``"mrr/95"`` or ``"hits@10/_hypernym"``). These are useful in the
#: on-disk eval dumps but must not be pushed to wandb: one eval epoch produces
#: thousands of them, and averaging them across datasets is meaningless.
_PER_RELATION_METRIC_RE = re.compile(r"^(?:mrr|hits@\d+|num_queries)/")


def is_per_relation_metric(metric_name: str) -> bool:
    """True for per-relation breakdown keys produced by :meth:`RankAccum.as_metrics`."""
    return bool(_PER_RELATION_METRIC_RE.match(metric_name))


def should_log_metric_to_wandb(
    metric_name: str, *, log_per_relation: bool | None = None
) -> bool:
    """Whether an eval metric leaf name should be uploaded to W&B.

    Always drops ``num_queries`` (aggregate and per-relation). Drops other
    per-relation breakdowns (``mrr/<id>``, ``hits@k/<id>``) unless
    ``WANDB_LOG_PER_RELATION=1`` / ``log_per_relation=True``.
    """
    if metric_name == "num_queries" or metric_name.startswith("num_queries/"):
        return False
    if is_per_relation_metric(metric_name):
        if log_per_relation is None:
            import os

            log_per_relation = os.environ.get("WANDB_LOG_PER_RELATION", "0") == "1"
        return bool(log_per_relation)
    return True



class RankAccum:
    """Running MRR / Hits@1/3/10, optionally grouped by relation id."""

    def __init__(self) -> None:
        self.rr = 0.0
        self.n = 0
        self.h1 = 0
        self.h3 = 0
        self.h10 = 0
        self.by_rel: dict[int, dict[str, float]] = defaultdict(
            lambda: {"rr": 0.0, "n": 0.0, "h1": 0.0, "h3": 0.0, "h10": 0.0}
        )

    def add(self, rank: int, rel_id: Optional[int] = None) -> None:
        if rank <= 0:
            return
        inv = 1.0 / float(rank)
        self.rr += inv
        self.n += 1
        self.h1 += int(rank <= 1)
        self.h3 += int(rank <= 3)
        self.h10 += int(rank <= 10)
        if rel_id is None:
            return
        bucket = self.by_rel[int(rel_id)]
        bucket["rr"] += inv
        bucket["n"] += 1.0
        bucket["h1"] += float(rank <= 1)
        bucket["h3"] += float(rank <= 3)
        bucket["h10"] += float(rank <= 10)

    def as_metrics(
        self, rel_names: Optional[Sequence[str]] = None
    ) -> Dict[str, float]:
        denom = max(self.n, 1)
        out: Dict[str, float] = {
            "mrr": self.rr / denom if self.n else 0.0,
            "hits@1": self.h1 / denom if self.n else 0.0,
            "hits@3": self.h3 / denom if self.n else 0.0,
            "hits@10": self.h10 / denom if self.n else 0.0,
            "num_queries": float(self.n),
        }
        for rel_id in sorted(self.by_rel):
            stats = self.by_rel[rel_id]
            n = int(stats["n"])
            d = max(n, 1)
            if rel_names is not None and 0 <= rel_id < len(rel_names):
                name = str(rel_names[rel_id])
            else:
                name = str(rel_id)
            out[f"mrr/{name}"] = stats["rr"] / d if n else 0.0
            out[f"hits@1/{name}"] = stats["h1"] / d if n else 0.0
            out[f"hits@3/{name}"] = stats["h3"] / d if n else 0.0
            out[f"hits@10/{name}"] = stats["h10"] / d if n else 0.0
            out[f"num_queries/{name}"] = float(n)
        return out

    def to_tensor(self, num_relations: int, device: torch.device) -> Tensor:
        """Pack overall + per-relation stats for ``all_reduce``. Shape ``[1+R, 5]``."""
        r = max(int(num_relations), 0)
        packed = torch.zeros(1 + r, 5, dtype=torch.float64, device=device)
        packed[0, 0] = self.rr
        packed[0, 1] = float(self.n)
        packed[0, 2] = float(self.h1)
        packed[0, 3] = float(self.h3)
        packed[0, 4] = float(self.h10)
        for rel_id, stats in self.by_rel.items():
            if rel_id < 0 or rel_id >= r:
                continue
            packed[1 + rel_id, 0] = stats["rr"]
            packed[1 + rel_id, 1] = stats["n"]
            packed[1 + rel_id, 2] = stats["h1"]
            packed[1 + rel_id, 3] = stats["h3"]
            packed[1 + rel_id, 4] = stats["h10"]
        return packed

    @classmethod
    def from_tensor(cls, packed: Tensor) -> "RankAccum":
        acc = cls()
        row0 = packed[0]
        acc.rr = float(row0[0].item())
        acc.n = int(row0[1].item())
        acc.h1 = int(row0[2].item())
        acc.h3 = int(row0[3].item())
        acc.h10 = int(row0[4].item())
        for rel_id in range(packed.shape[0] - 1):
            row = packed[1 + rel_id]
            n = float(row[1].item())
            if n <= 0:
                continue
            acc.by_rel[rel_id] = {
                "rr": float(row[0].item()),
                "n": n,
                "h1": float(row[2].item()),
                "h3": float(row[3].item()),
                "h10": float(row[4].item()),
            }
        return acc


def relation_names_from_bundle(bundle) -> Optional[list[str]]:
    names = getattr(bundle, "metadata", None) or {}
    raw = names.get("relation_names") if isinstance(names, Mapping) else None
    if not raw:
        return None
    return [str(x) for x in raw]
