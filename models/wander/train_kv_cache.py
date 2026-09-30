"""Cached per-pass train-node hidden states for the cached-KV NC eval path.

Phase A (``Wander.precompute_train_kv_cache``) runs train-chunk walk passes and
snapshots the hidden states of all train nodes right before the global-attention
block of every refinement step. Walk starts are stratified over chunks of train
nodes (``train_kv_chunk_size``); query ``batch_indices`` stays empty so chunk
nodes are not treated as eval batch nodes. Phase B replays one cached pass per
pass per ensemble draw: the snapshots are projected through the matching
``net_global`` layer's ``attn_norm`` + ``wk`` / ``wv`` once per draw and used
as external K/V for global attention. Query nodes in the current batch are
then dropped from that K/V (``exclude_nodes``) so train_eval matches the live
path's ``train_mask & ~in_batch``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
import torch.distributed as dist

_WARN_BYTES = 8 * 1024 ** 3  # warn above 8 GB of CPU cache


@dataclass
class TrainKVPass:
    """Snapshots of one Phase A pass.

    ``hidden[t]`` maps channel name -> CPU tensor:
      - ``"node"``:    [M, D]
      - ``"feature"``: [M, F, D] (absent when the graph has no features)
      - ``"label"``:   [M, C, D]
    captured right before the global-attention block of refinement step ``t``.
    """

    feat_perm: Optional[Tensor]
    label_perm: Optional[Tensor]
    hidden: List[Dict[str, Tensor]]


# Per refinement step: channel name -> (k, v), shaped for direct use as
# ``external_kv`` in the global-attention nets:
#   "node":    ([1, M, D], [1, M, D])
#   "feature": ([F, M, D], [F, M, D])
#   "label":   ([C, M, D], [C, M, D])
MaterializedPass = List[Dict[str, Tuple[Tensor, Tensor]]]


@dataclass
class TrainKVCache:
    """CPU-resident train-node hidden-state cache with per-pass materialization."""

    train_ids: Tensor  # [M] global node ids (CPU, long)
    passes: List[TrainKVPass] = field(default_factory=list)

    _materialized_idx: Optional[int] = field(default=None, repr=False)
    _materialized: Optional[MaterializedPass] = field(default=None, repr=False)

    @property
    def num_passes(self) -> int:
        return len(self.passes)

    @property
    def num_train_nodes(self) -> int:
        return int(self.train_ids.numel())

    def size_bytes(self) -> int:
        total = 0
        for p in self.passes:
            for step in p.hidden:
                for t in step.values():
                    total += t.numel() * t.element_size()
        return total

    def add_pass(
        self,
        feat_perm: Optional[Tensor],
        label_perm: Optional[Tensor],
        hidden: List[Dict[str, Tensor]],
    ) -> None:
        self.passes.append(
            TrainKVPass(
                feat_perm=feat_perm.detach().cpu() if feat_perm is not None else None,
                label_perm=label_perm.detach().cpu() if label_perm is not None else None,
                hidden=hidden,
            )
        )

    def log_size(self) -> None:
        gb = self.size_bytes() / 1024 ** 3
        msg = (
            f"[train_kv_cache] {self.num_passes} passes x "
            f"{self.num_train_nodes} train nodes: {gb:.2f} GB (CPU)"
        )
        if self.size_bytes() > _WARN_BYTES:
            msg += "  WARNING: large cache; consider fewer passes or bf16 storage."
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(msg)

    @torch.no_grad()
    def materialize(
        self, pass_idx: int, model, device: torch.device,
    ) -> MaterializedPass:
        """Project pass ``pass_idx`` through ``net_global`` K/V weights on ``device``.

        Memoizes a single pass at a time (draws within one batch iterate passes
        sequentially; re-materialization per draw is cheap relative to the
        forward pass).
        """
        if self._materialized_idx == pass_idx and self._materialized is not None:
            return self._materialized
        self.release_materialized()

        cached_pass = self.passes[pass_idx]
        out: MaterializedPass = []
        for t, step in enumerate(cached_pass.hidden):
            t_inter = 0 if model.parameter_tying_across_refinements else t
            step_kv: Dict[str, Tuple[Tensor, Tensor]] = {}
            for name, h in step.items():
                if name == "node":
                    ch_idx = 0
                elif name == "feature":
                    ch_idx = 0 if model.parameter_tying_across_channels else 1
                else:  # "label"
                    ch_idx = 0 if model.parameter_tying_across_channels else 2
                net = model._net_global_for_task("node")[t_inter][ch_idx]
                assert len(net.layers) == 1, (
                    "cached train KV requires wander_n_layers == 1"
                )
                layer = net.layers[0]
                h_dev = h.to(device=device, dtype=layer.wk.weight.dtype)
                if name == "node":
                    # [M, D] -> [1, M, D]
                    h_rows = h_dev[None]
                else:
                    # [M, K, D] -> [K, M, D] (one attention row per channel)
                    h_rows = h_dev.permute(1, 0, 2).contiguous()
                normed = layer.attn_norm(h_rows)
                step_kv[name] = (layer.wk(normed), layer.wv(normed))
            out.append(step_kv)

        self._materialized_idx = pass_idx
        self._materialized = out
        return out

    @staticmethod
    @torch.no_grad()
    def exclude_nodes(
        materialized: MaterializedPass,
        train_ids: Tensor,
        exclude_ids: Tensor,
    ) -> MaterializedPass:
        """Drop ``exclude_ids`` from the train-node (M) axis of ``materialized``.

        Matches live global-attention eval (``train_mask & ~in_batch``): query
        nodes must not appear as labeled K/V, including when evaluating on the
        train split itself. Does not mutate the memoized full-pass tensors.
        """
        if exclude_ids.numel() == 0:
            return materialized
        ids = train_ids.to(device=exclude_ids.device)
        keep = ~torch.isin(ids, exclude_ids)
        assert bool(keep.any().item()), (
            "cached train KV is empty after excluding batch nodes; "
            "the batch covers the entire train set"
        )
        # Index on the device of the materialized tensors (usually CUDA).
        device = next(iter(materialized[0].values()))[0].device
        keep_idx = keep.nonzero(as_tuple=True)[0].to(device=device)

        out: MaterializedPass = []
        for step_kv in materialized:
            filtered: Dict[str, Tuple[Tensor, Tensor]] = {}
            for name, (k, v) in step_kv.items():
                # node: [1, M, D]; feature/label: [K, M, D]
                filtered[name] = (k.index_select(1, keep_idx), v.index_select(1, keep_idx))
            out.append(filtered)
        return out

    def release_materialized(self) -> None:
        self._materialized_idx = None
        self._materialized = None

    def release(self) -> None:
        """Drop everything (call when a dataset's eval is finished)."""
        self.release_materialized()
        self.passes = []
