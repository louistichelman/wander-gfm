from typing import Any, Callable, Dict, List, Optional, Tuple
from contextlib import contextmanager
import os
import time
import numpy as np
import torch
from torch import nn, Tensor
import torch.distributed as dist
from torch._refs import T
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
import math

import graph_walker

from .ensemble_prefetch import (
    AsyncWalkTransfer,
    EnsemblePrefetchStats,
    WalkPrefetcher,
    run_pipelined_ensemble,
)
from .nc_train_query import nc_train_query_count, sample_nc_train_query_ids
from .train_kv_cache import TrainKVCache
from .walk_cache import get_rw_profile, get_walk_graph_cache
from .consensus import consensus_mean, consensus_softmax_stable
from .embeddings import LabelEmbedding, FeatureEmbedding
from .graph_visit_stats import original_graph_visit_stats as _original_graph_visit_stats
from .sublayers import (
    RMSNorm, BidirectionalGRU, materialize_if_broadcast,
    SequenceAttention, IntraNodeAttention,
    set_kernel_compile,
)

torch.set_float32_matmul_precision("high")

# cuDNN SDPA backend fails during gradient-checkpoint recomputation; flash
# SDPA can reject large batch dims. Disable both so PyTorch uses the
# memory-efficient (xformers) backend or math fallback.
# torch.backends.cuda.enable_cudnn_sdp(False)
# torch.backends.cuda.enable_flash_sdp(False)


def to_csr_tensor(edge_index: Tensor, edge_type: Tensor, num_nodes=None):
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1
    # get source and destination nodes
    src = edge_index[0]
    dst = edge_index[1]
    # sort edges by source node, then destination node
    # this is crucial for proper CSR format
    sort_idx = torch.argsort(src * num_nodes + dst)
    src_sorted = src[sort_idx]
    dst_sorted = dst[sort_idx]
    edge_type_sorted = edge_type[sort_idx]
    # count edges per source node
    src_counts = torch.bincount(src_sorted, minlength=num_nodes)
    # build indptr (crow_indices)
    indptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=edge_index.device)
    indptr[1:] = torch.cumsum(src_counts, dim=0)
    # convert to numpy arrays with specified dtype
    indptr_np = indptr.cpu().numpy().astype(np.uint32)
    indices_np = dst_sorted.cpu().numpy().astype(np.uint32)
    data_np = edge_type_sorted.cpu().numpy().astype(np.uint32)
    return indptr_np, indices_np, data_np


def balanced_sample(
    x: Tensor, x_types: Tensor, num_types: int, samples_per_type: int
) -> Tuple[Tensor, Tensor]:
    # num_types must be greater than or equal to unique type count shown in x_types
    # if num_types is smaller, than it means some types does not exist in x_types
    # assert num_types >= len(torch.unique(x_types)), "Number of types must be greater than or equal to unique type count shown in x_types"
    sampled_indices = []
    for i in range(num_types):
        # get indices for the current type
        type_indices = (x_types == i).nonzero(as_tuple=True)[0]
        # if type does not exist in x_types, skip it
        if len(type_indices) == 0:
            continue
        # randomly choose indices for this type
        num_repeats = (samples_per_type + len(type_indices) - 1) // len(type_indices)
        rand_indices = torch.cat(
            [
                torch.randperm(len(type_indices), device=x.device)
                for _ in range(num_repeats)
            ]
        )[:samples_per_type]
        assert len(rand_indices) == samples_per_type
        # collect sampled indices
        sampled_indices.append(type_indices[rand_indices])
    # concatenate indices
    indices = torch.cat(sampled_indices)
    return x.t()[indices].contiguous(), x_types[indices].contiguous()


_FEATURE_WALK_NUM_CAP_THRESHOLD = 16
_FEATURE_WALK_NUM_CAP = 64


def _uniform_train_starts(
    train_idx: Tensor,
    n_starts: int,
    num_groups: int,
    device: torch.device,
) -> Tensor:
    num_train = int(train_idx.shape[0])
    pick = torch.randint(num_train, (n_starts, num_groups), device=device)
    return train_idx[pick]


def _nc_combined_walk_start_count(eff_N: int, sample_T: int) -> int:
    """Total NC walk starts for one forward (train+batch budget, no short_random doubling).

    Yields ``walks_per_sample ≈ 3 * eff_N`` per refinement step.
    """
    return math.floor(1.5 * eff_N * sample_T) + math.ceil(1.5 * eff_N * sample_T)


def _nc_walk_start_counts(
    eff_N: int, sample_T: int, short_random_starts: bool,
) -> tuple[int, int, int]:
    """Return ``(n_random_starts, n_batch_starts, walks_per_sample)``."""
    n_random_base = math.floor(1.5 * eff_N * sample_T)
    n_batch_starts = math.ceil(1.5 * eff_N * sample_T)
    n_random_starts = (2 * n_random_base) if short_random_starts else n_random_base
    total_start_rows = n_random_starts + n_batch_starts
    walks_per_sample = total_start_rows // sample_T
    return n_random_starts, n_batch_starts, walks_per_sample


def _stratified_batch_starts(
    batch_indices: Tensor, walks_per_sample: int, sample_T: int,
    device: torch.device,
) -> Tensor:
    """Stratified walk-start assignment over batch nodes, per refinement step.

    Returns a flat [walks_per_sample * sample_T] tensor laid out so that row
    ``r`` maps to (slot ``r // sample_T``, refinement ``r % sample_T``) —
    matching the ``view(walks_per_sample, sample_T, ...)`` reshape downstream.
    For every refinement step, each batch node receives between
    ``floor(walks_per_sample / K)`` and ``ceil(walks_per_sample / K)`` starts
    (fully vectorized; no per-node loop).
    """
    K = int(batch_indices.numel())
    assert K > 0
    n_cycles = math.ceil(walks_per_sample / K)
    # Independent random permutation of the batch per (refinement, cycle).
    perms = torch.argsort(torch.rand(sample_T, n_cycles, K, device=device), dim=2)
    tiled = batch_indices.to(device)[perms].reshape(sample_T, n_cycles * K)
    tiled = tiled[:, :walks_per_sample]  # [sample_T, walks_per_sample]
    return tiled.transpose(0, 1).reshape(-1)


def _as_walk_torch(channel: Tensor | np.ndarray) -> Tensor:
    """Normalize walk channels to CPU torch tensors for padding/concat."""
    return channel if isinstance(channel, Tensor) else torch.as_tensor(channel)


def _pad_walk_channel(channel: Tensor | np.ndarray, target_len: int, *, pad_restarts: bool) -> Tensor:
    """Pad walk channels along the last dimension to ``target_len``."""
    channel = _as_walk_torch(channel)
    src_len = int(channel.shape[-1])
    if src_len >= target_len:
        return channel[..., :target_len]
    pad_len = target_len - src_len
    if pad_restarts:
        pad_val = 0
    else:
        pad_val = channel[..., -1:]
        pad_val = pad_val.expand(*channel.shape[:-1], pad_len)
        return torch.cat([channel, pad_val], dim=-1)
    pad_shape = (*channel.shape[:-1], pad_len)
    padding = torch.full(pad_shape, pad_val, dtype=channel.dtype, device=channel.device)
    return torch.cat([channel, padding], dim=-1)


def _pad_walk_tuple(walk_tuple: tuple, target_len: int) -> tuple:
    walks, named_walks, restarts, neighbors, types, named_types, directions = walk_tuple
    return (
        _pad_walk_channel(walks, target_len, pad_restarts=False),
        _pad_walk_channel(named_walks, target_len, pad_restarts=False),
        _pad_walk_channel(restarts, target_len, pad_restarts=True),
        _pad_walk_channel(neighbors, target_len, pad_restarts=False),
        _pad_walk_channel(types, target_len, pad_restarts=False),
        _pad_walk_channel(named_types, target_len, pad_restarts=False),
        _pad_walk_channel(directions, target_len, pad_restarts=False),
    )


def _concat_walk_tuples(random_tuple: tuple, batch_tuple: tuple) -> tuple:
    return tuple(
        torch.cat([_as_walk_torch(random_part), _as_walk_torch(batch_part)], dim=0)
        for random_part, batch_part in zip(random_tuple, batch_tuple)
    )


def _filter_train_free_walks(
    walk_tuple: tuple,
    train_mask: Tensor,
    keep_prob: float,
) -> tuple:
    """Keep train-hitting walks always; keep train-free walks with probability p."""
    walks = walk_tuple[0]
    if not isinstance(walks, Tensor) or walks.ndim != 4:
        return walk_tuple
    n_layers, n_graphs, n_walks, _walk_len = walks.shape
    tm = train_mask.to(device=walks.device, dtype=torch.bool)
    n_nodes = int(tm.numel())
    valid = (walks >= 0) & (walks < n_nodes)
    safe = walks.clamp(min=0, max=max(n_nodes - 1, 0))
    hits = (tm[safe] & valid).any(dim=-1)
    if keep_prob >= 1.0:
        return walk_tuple

    keep = hits.clone()
    if keep_prob > 0.0:
        rand = torch.rand(hits.shape, device=walks.device)
        keep = hits | ((~hits) & (rand < keep_prob))

    keep_idx: list[list[Tensor]] = []
    n_kept: list[int] = []
    for t in range(n_layers):
        row: list[Tensor] = []
        for b in range(n_graphs):
            idx = keep[t, b].nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                idx = torch.zeros(1, dtype=torch.long, device=walks.device)
            row.append(idx)
            n_kept.append(int(idx.numel()))
        keep_idx.append(row)
    max_n = max(n_kept)

    filtered: list[Tensor] = []
    for ch in walk_tuple:
        if not isinstance(ch, Tensor) or ch.ndim < 3:
            filtered.append(ch)
            continue
        parts_t = []
        for t in range(n_layers):
            parts_b = []
            for b in range(n_graphs):
                sl = ch[t, b].index_select(0, keep_idx[t][b])
                if sl.shape[0] < max_n:
                    pad_n = max_n - sl.shape[0]
                    sl = torch.cat([sl, sl[-1:].expand(pad_n, *sl.shape[1:])], dim=0)
                parts_b.append(sl)
            parts_t.append(torch.stack(parts_b, dim=0))
        filtered.append(torch.stack(parts_t, dim=0))
    return tuple(filtered)


# PyG Data attributes indexed by global node id (first dim == num_nodes); subset with keep_old when pruning.
_NODE_ALIGNED_OPTIONAL_KEYS = ("train_mask", "val_mask", "test_mask", "graph_id")


class Wander(nn.Module):
    @staticmethod
    def _supervise_non_train_nodes(data: Data) -> bool:
        """True when training loss should cover all held-out (val|test) nodes.

        An explicit ``data.supervise_non_train_nodes`` flag (True or False)
        takes precedence.
        """
        flag = getattr(data, "supervise_non_train_nodes", None)
        if flag is not None:
            return bool(flag)
        val_mask = getattr(data, "val_mask", None)
        if val_mask is None:
            return True
        return not val_mask.any()

    @staticmethod
    def _non_train_supervise_idx(data: Data, num_graph_nodes: Optional[int] = None) -> Tensor:
        """Indices of held-out nodes (val ∪ test) for synthetic-style NC loss.

        Prefer ``val_mask | test_mask`` when both exist; fall back to ``test_mask``
        alone (synthetic graphs typically have no val). ``num_graph_nodes``
        truncates to the surviving RW-pruned prefix when set.
        """
        n = int(data.num_nodes) if num_graph_nodes is None else int(num_graph_nodes)
        device = data.y.device if getattr(data, "y", None) is not None else (
            data.test_mask.device if getattr(data, "test_mask", None) is not None
            else torch.device("cpu")
        )
        out = torch.zeros(n, dtype=torch.bool, device=device)
        test_mask = getattr(data, "test_mask", None)
        if test_mask is not None:
            out = out | test_mask[:n].to(device=device)
        val_mask = getattr(data, "val_mask", None)
        if val_mask is not None:
            out = out | val_mask[:n].to(device=device)
        return out.nonzero(as_tuple=True)[0]

    def _apply_feature_column_randomization(
        self, features: Tensor, perm: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Optionally permute raw feature columns; returns ``(features, perm)``.

        ``perm`` replays a previously sampled permutation (cached-KV eval).
        """
        if not self.randomize_feat_columns:
            return features, None
        f_raw = int(features.shape[1])
        if f_raw < 1:
            return features, None
        if perm is None:
            perm = torch.randperm(f_raw, device=features.device)
        else:
            perm = perm.to(device=features.device)
        return features[:, perm], perm

    def _permute_label_columns(
        self, y: Tensor, perm: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Optionally permute label class columns; returns ``(y, label_perm)``.

        ``perm`` replays a previously sampled permutation (cached-KV eval).
        """
        num_classes = int(y.shape[-1])
        device = y.device
        if perm is not None:
            label_perm = perm.to(device=device)
        elif not self.randomize_label_columns or num_classes < 2:
            label_perm = torch.arange(num_classes, device=device)
        else:
            label_perm = torch.randperm(num_classes, device=device)
        return y[:, label_perm], label_perm

    @staticmethod
    def _data_has_real_features(data: Data) -> bool:
        x = getattr(data, "x", None)
        return x is not None and isinstance(x, Tensor) and x.ndim == 2 and x.shape[1] > 0

    @staticmethod
    def _data_has_labels(data: Data) -> bool:
        y = getattr(data, "y", None)
        return y is not None and isinstance(y, Tensor) and y.ndim == 2 and y.shape[1] > 0

    def _needs_per_draw_randomization(self, data: Data) -> bool:
        return (
            (self.randomize_feat_columns and self._data_has_real_features(data))
            or (self.randomize_label_columns and self._data_has_labels(data))
        )

    def _finalize_cpu_walk_tuple(self, walk_tuple: tuple, num_types: int) -> tuple:
        """Convert walk channels to CPU long tensors and apply sentinel indices."""
        walks, named_walks, restarts, neighbors, types, named_types, directions = walk_tuple
        out = []
        for arr in (walks, named_walks, restarts, neighbors, types, named_types, directions):
            if isinstance(arr, Tensor):
                out.append(arr.to(device=torch.device("cpu"), dtype=torch.long))
            else:
                out.append(torch.tensor(arr.astype(np.int32), dtype=torch.long))
        walks, named_walks, restarts, neighbors, types, named_types, directions = out
        types = types.clone()
        types[types == -1] = num_types
        named_types = named_types.clone()
        named_types[named_types == -1] = self.L_max + 1
        directions = directions.clone()
        directions[directions == -1] = 3
        return walks, named_walks, restarts, neighbors, types, named_types, directions

    def _walk_tuple_to_device(
        self,
        walk_tuple: tuple,
        device: torch.device,
        num_types: int,
    ) -> tuple:
        finalized = self._finalize_cpu_walk_tuple(walk_tuple, num_types)
        if device.type == "cuda":
            return AsyncWalkTransfer._to_device(finalized, device, non_blocking=False)
        return tuple(t.to(device=device) for t in finalized)

    def _pipelined_ensemble_from_walks(
        self,
        generate_fn: Callable[[], tuple],
        device: torch.device,
        forward_from_walks: Callable[[tuple], Any],
    ) -> list:
        stats = EnsemblePrefetchStats()
        prefetcher = WalkPrefetcher(
            generate_fn,
            self.test_samples,
            queue_depth=self.ensemble_prefetch_queue_depth,
            stats=stats,
        )
        outputs, stats = run_pipelined_ensemble(
            prefetcher,
            device,
            forward_from_walks,
            self.test_samples,
        )
        self._last_ensemble_prefetch_stats = stats
        return outputs

    def _concat_thinking_features(
        self, h_feature: Tensor, bsize: int, num_graph_nodes: int,
    ) -> Tensor:
        """Append learnable thinking-feature slots after real features on graph nodes."""
        if self.thinking_features == 0:
            return h_feature
        think = self.thinking_feature_emb
        if self.feature_init_norm is not None:
            think = self.feature_init_norm(think)
        think_exp = think[None, None, :, :].expand(
            bsize, num_graph_nodes, self.thinking_features, self.D
        )
        if not isinstance(h_feature, Tensor) or h_feature.ndim != 4:
            return think_exp
        return torch.cat([h_feature, think_exp], dim=2)

    def _append_thinking_row_nodes(
        self,
        h_node: Tensor,
        h_feature: Tensor,
        h_label: Tensor,
        bsize: int,
        num_features: int,
        has_features: bool,
        has_labels: bool,
        num_label_tokens: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Append edgeless virtual nodes whose tokens are only the row embedding."""
        if self.thinking_rows == 0:
            return h_node, h_feature, h_label
        node_rows = []
        feat_rows = []
        lab_rows = []
        for i in range(self.thinking_rows):
            row = self.thinking_row_emb[i]
            node_rows.append(row.view(1, 1, self.D).expand(bsize, 1, self.D))
            if has_features:
                feat_rows.append(
                    row.view(1, 1, 1, self.D).expand(bsize, 1, num_features, self.D)
                )
            if has_labels:
                lab_rows.append(
                    row.view(1, 1, 1, self.D).expand(
                        bsize, 1, num_label_tokens, self.D
                    )
                )
        h_node = torch.cat([h_node, *node_rows], dim=1)
        if has_features:
            h_feature = torch.cat([h_feature, *feat_rows], dim=1)
        if has_labels:
            h_label = torch.cat([h_label, *lab_rows], dim=1)
        return h_node, h_feature, h_label

    @staticmethod
    def _extend_train_mask_for_thinking_rows(
        train_mask: Tensor, num_thinking_rows: int, device: torch.device,
    ) -> Tensor:
        if num_thinking_rows == 0:
            return train_mask
        extra = torch.ones(num_thinking_rows, dtype=torch.bool, device=device)
        return torch.cat([train_mask, extra])

    @staticmethod
    def _tensor_storage_numel(x: Tensor) -> int:
        try:
            return int(x.untyped_storage().size() // max(x.element_size(), 1))
        except Exception:
            return -1

    @classmethod
    def _materialize_token_states(
        cls,
        h_node: Tensor,
        h_feature: Tensor,
        h_label: Tensor,
        has_features: bool,
        has_labels: bool,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        h_node = materialize_if_broadcast(h_node)
        if has_features and isinstance(h_feature, Tensor) and h_feature.ndim >= 2:
            h_feature = materialize_if_broadcast(h_feature)
        if has_labels and isinstance(h_label, Tensor) and h_label.ndim >= 2:
            h_label = materialize_if_broadcast(h_label)
        return h_node, h_feature, h_label

    def _log_wander_mem(
        self,
        *,
        num_nodes: int,
        has_features: bool,
        has_labels: bool,
        h_node: Tensor,
        h_feature: Tensor,
        h_label: Tensor,
        global_kv_mask: Optional[Tensor],
        tag: str = "",
    ) -> None:
        if os.environ.get("LOG_WANDER_MEM", "0") != "1":
            return
        k = int(h_feature.shape[2]) if has_features and isinstance(h_feature, Tensor) and h_feature.ndim == 4 else 0
        c = int(h_label.shape[2]) if has_labels and isinstance(h_label, Tensor) and h_label.ndim == 4 else 0
        kv_m = (
            int(global_kv_mask.sum(dim=1).max().item())
            if global_kv_mask is not None
            else num_nodes
        )
        peak = (
            torch.cuda.max_memory_allocated() / 1e9
            if torch.cuda.is_available()
            else 0.0
        )
        print(
            f"[wander_mem{(' ' + tag) if tag else ''}] intra={self.use_intra_node_attention} "
            f"N={num_nodes} kv_M={kv_m} K={k} C={c} "
            f"global_channel_chunk={self.global_channel_chunksize} "
            f"h_node.stride={tuple(h_node.stride())} "
            f"h_node.contiguous={h_node.is_contiguous()} "
            f"h_node.storage_numel={self._tensor_storage_numel(h_node)} "
            f"h_node.numel={h_node.numel()} "
            f"        peak_gb={peak:.2f}",
            flush=True,
        )

    def __init__(self, model_cfg: dict):
        super().__init__()

        # inference settings
        self.test_samples = model_cfg.get("test_samples", 1)
        self.use_ensemble_prefetch = model_cfg.get("use_ensemble_prefetch", True)
        self.ensemble_prefetch_queue_depth = int(
            model_cfg.get("ensemble_prefetch_queue_depth", 2)
        )
        self._last_ensemble_prefetch_stats: Optional[EnsemblePrefetchStats] = None
        self.randomize_feat_columns = model_cfg.get("randomize_feat_columns", False)
        self.randomize_label_columns = model_cfg.get("randomize_label_columns", False)
        self.nc_train_query_frac = float(model_cfg.get("nc_train_query_frac", 0.0) or 0.0)
        self.nc_train_query_cap = int(model_cfg.get("nc_train_query_cap", 0) or 0)

        # Cached train-KV eval (two-phase NC eval). The cache itself is
        # attached per dataset by the experiment via attach_train_kv_cache().
        self.cached_train_kv = model_cfg.get("cached_train_kv", False)
        self.train_kv_passes = int(model_cfg.get("train_kv_passes", 16))
        self.train_kv_chunk_size = int(model_cfg.get("train_kv_chunk_size", 512))
        self.train_kv_walk_num = int(
            model_cfg.get("train_kv_walk_num", model_cfg["walk_num"])
        )
        eval_walk_num = model_cfg.get("eval_walk_num")
        self.eval_walk_num = (
            int(eval_walk_num) if eval_walk_num is not None else None
        )
        keep_p = model_cfg.get("keep_train_free_p")
        self.keep_train_free_p = None if keep_p is None else float(keep_p)
        self.train_kv_cache_dtype = getattr(
            torch, model_cfg.get("train_kv_cache_dtype", "float32")
        )
        self._train_kv_cache: Optional[TrainKVCache] = None

        # random walk
        self.N = model_cfg["walk_num"]
        self.L = model_cfg["walk_len"]
        self.L_max = model_cfg["max_walk_len"]
        self.adaptive_walks = model_cfg.get("adaptive_walks", False)
        self.record_neighbors = model_cfg["record_neighbors"]
        self.fast_uniform_walks = bool(model_cfg.get("fast_uniform_walks", False))
        # torch.compile RMSNorm / SwiGLU / GRU residual-FFN during eval.
        self.compile_eval = bool(model_cfg.get("compile_eval", False))
        self.use_walk_cache = model_cfg.get("use_walk_cache", True)
        self.fixed_rw_across_refinements = model_cfg.get("fixed_rw_across_refinements", False)
        self.concat_walks = model_cfg.get("concat_walks", False)

        self.short_random_starts = model_cfg.get("short_random_starts", False)

        
        # refinements
        self.T = model_cfg["refinements"]
        self.attention_scatter = model_cfg["attention_scatter"]
        self.H = model_cfg["attention_scatter_n_heads"]
        self.additive_refinement = model_cfg["additive_refinement"]
        self.use_init_norm = model_cfg.get("init_norm", True)
        # Outer Pre-LN only applies to additive RW residuals.
        self.use_pre_norm = bool(model_cfg.get("pre_norm", False)) and self.additive_refinement
        self.embed_first_only = model_cfg["embed_only_first_refinement"]
        self.embedding_tying = model_cfg["embedding_tying_across_refinements"]
        self.parameter_tying_across_refinements = model_cfg["parameter_tying_across_refinements"]
        self.parameter_tying_across_channels = model_cfg["parameter_tying_across_channels"]
        self.untie_attention_across_tasks = model_cfg.get(
            "untie_attention_across_tasks", False
        )
        
        # intra-node attention
        self.use_intra_node_attention = model_cfg.get("intra_node_attention", False)
        self.intra_node_attention_n_heads = model_cfg.get("intra_node_attention_n_heads", 1)
        self.rotary_emb_intra_node_attention = model_cfg.get(
            "rotary_emb_intra_node_attention", False
        )
        self.intra_node_ffn = bool(model_cfg.get("intra_node_ffn", False))

        # thinking features / rows
        self.thinking_features = model_cfg.get("thinking_features", 0)
        self.thinking_rows = model_cfg.get("thinking_rows", 0)

        # inter-node chunking for feature/label channels (sequence nets + local attention)
        self.inter_node_chunksize = model_cfg.get(
            "inter_node_chunksize", model_cfg.get("gru_chunk_size", None)
        )

        # gradient checkpointing
        self.checkpoint_refinements = model_cfg.get("checkpoint_refinements", True)
        self.checkpoint_sublayers = model_cfg.get(
            "checkpoint_sublayers", model_cfg.get("checkpoint_sequence_net", True)
        )

        # intra-node attention chunking (chunk over B*N)
        self.intra_node_chunksize = model_cfg.get(
            "intra_node_chunksize", model_cfg.get("intra_attn_chunk_size", 512)
        )
        # Global ICL over feature/label channels is batched as B*C sequences.
        # FULL_CORA C=70 + math SDPA (forced by a padding mask) is ~27GB per
        # refinement. Chunk so at most this many channels share one SDPA.
        env_chunk = os.environ.get("WANDER_GLOBAL_CHANNEL_CHUNK", "").strip()
        self.global_channel_chunksize = int(
            env_chunk
            if env_chunk
            else model_cfg.get("global_channel_chunksize", 8)
        )

        # LayerScale for additive refinement updates (0 = disabled)
        self.layerscale_init = model_cfg.get("layerscale", 0.0)

        # global attention update (in context learning)
        self.global_attention_update = model_cfg.get("global_attention_update", False)
        # None = keep all query-relation neighbors as LP global-attention K/V.
        self.global_kv_neighbors = model_cfg.get("global_kv_neighbors", None)

        # random-walk based inter-node update
        self.rw_update = model_cfg.get("rw_update", True)

        self.add_inverse_edges_kgs = model_cfg.get("add_inverse_edges_kgs", True)
        # When inverse edges are added: if True, the walk parser only uses forward
        # (along-walk) edges and the direction embedding is dropped; if False
        # (default), use the random forward/reverse candidate pick plus the
        # direction embedding.
        self.only_forward_edges_when_inverses_added = model_cfg.get(
            "wander_only_forward_edges_when_inverses_added", False
        )
        self.disable_rw_graph_prune = model_cfg.get("disable_rw_graph_prune", False)

        # hidden states
        self.D = model_cfg["hidden_dim"]
        self.dtype = getattr(torch, model_cfg["dtype"])

        # node and type embeddings
        # Role-based node-token initialization: 0=other, 1=query-relation-neighbor,
        # 2=query. The role-0 row is the shared default init (used for every node in
        # node classification); link prediction uses all three rows.
        self.node_role_emb = nn.Embedding(3, self.D).to(self.dtype)
        self.type_init = nn.Parameter(torch.randn(self.D, dtype=self.dtype))

        # feature embeddings
        self.feature_groups = model_cfg.get("feature_groups", 1)
        self.feature_embedding = FeatureEmbedding(
            self.D,
            feature_groups=self.feature_groups,
            use_mlp=model_cfg.get("feature_embedding_mlp", False),
        ).to(self.dtype)
        self.feature_init_norm = (
            RMSNorm(self.D).to(self.dtype) if self.use_init_norm else None
        )
        if self.thinking_features > 0:
            self.thinking_feature_emb = nn.Parameter(
                torch.randn(self.thinking_features, self.D, dtype=self.dtype)
            )
        if self.thinking_rows > 0:
            self.thinking_row_emb = nn.Parameter(
                torch.randn(self.thinking_rows, self.D, dtype=self.dtype)
            )
        
        # label embeddings
        self.label_embedding = LabelEmbedding(2, self.D).to(self.dtype)
        self.label_init_norm = (
            RMSNorm(self.D).to(self.dtype) if self.use_init_norm else None
        )

        # walk embedding
        self.emb_anon_node = nn.ModuleList()
        self.emb_anon_type = nn.ModuleList()
        if self.record_neighbors:
            self.emb_neighbor = nn.ModuleList()
        self.emb_direction = nn.ModuleList()
        self.emb_node_is_query = nn.ModuleList()
        self.emb_type_is_query = nn.ModuleList()

        # sequence networks: dedicated module list per inter-node update path.
        # Both are always instantiated regardless of which flags are on.
        # - net_global: always SequenceAttention (rotary off) — used by the
        #   global-attention update path; ignores --wander_net.
        #   When untie_attention_across_tasks: net_global = node-cls, and
        #   net_global_link is a separate twin for link prediction.
        # - net_rw: SequenceAttention (rotary on) or BidirectionalGRU per
        #   --wander_net — used by the random-walk update path.
        self.net_global = nn.ModuleList()
        self.net_rw = nn.ModuleList()
        if self.untie_attention_across_tasks:
            self.net_global_link = nn.ModuleList()

        self.from_node = nn.ModuleList()
        self.from_feature = nn.ModuleList()
        self.from_label = nn.ModuleList()
        self.from_type = nn.ModuleList()

        self.to_node = nn.ModuleList()
        self.to_type = nn.ModuleList()
        self.to_feature = nn.ModuleList()
        self.to_label = nn.ModuleList()

        self.node_logit = nn.ModuleList()
        self.type_logit = nn.ModuleList()
        self.feature_logit = nn.ModuleList()
        self.label_logit = nn.ModuleList()

        for _ in range(self.T if not self.embedding_tying else 1):
            # anon_type     -1 (=L_max) no type
            # *_is_query    0 not query, 1 query, 2 no conditioning
            # neighbor      0 walk, 1 neighbor
            # direction     0 downstream, 1 upstream, 2 loop, 3 no direction
            self.emb_anon_node.append(nn.Embedding(self.L_max, self.D).to(self.dtype))
            self.emb_anon_type.append(nn.Embedding(self.L_max + 1, self.D).to(self.dtype))
            if self.record_neighbors:
                self.emb_neighbor.append(nn.Embedding(2, self.D).to(self.dtype))
            self.emb_direction.append(nn.Embedding(4, self.D).to(self.dtype))
            self.emb_node_is_query.append(nn.Embedding(3, self.D).to(self.dtype))
            self.emb_type_is_query.append(nn.Embedding(3, self.D).to(self.dtype))

        n_heads = model_cfg.get("net_n_heads", max(1, self.D // 64))
        n_channels = 3 if not self.parameter_tying_across_channels else 1  # node, feature, label
        for _ in range(self.T if not self.parameter_tying_across_refinements else 1):
            self.net_global.append(
                self._make_global_refine_net(n_channels, n_heads, model_cfg)
            )
            if self.untie_attention_across_tasks:
                self.net_global_link.append(
                    self._make_global_refine_net(n_channels, n_heads, model_cfg)
                )

            rw_refine_net = nn.ModuleList()
            for _ in range(n_channels):
                if model_cfg["net"] == "gru":
                    rw_refine_net.append(
                        BidirectionalGRU(
                            dim=self.D,
                            n_layers=model_cfg["n_layers"],
                            multiple_of=self.D,
                            norm_eps=1e-5,
                        ).to(self.dtype)
                    )
                elif model_cfg["net"] == "attention":
                    rw_refine_net.append(
                        SequenceAttention(
                            dim=self.D,
                            n_layers=model_cfg["n_layers"],
                            n_heads=n_heads,
                            multiple_of=self.D,
                            norm_eps=1e-5,
                            use_rotary_embedding=True,
                        ).to(self.dtype)
                    )
                else:
                    raise NotImplementedError(
                        f"Network {model_cfg['net']} is not implemented"
                    )
            self.net_rw.append(rw_refine_net)

            self.from_node.append(nn.Linear(self.D, self.D).to(self.dtype))
            self.from_feature.append(nn.Linear(self.D, self.D).to(self.dtype))
            self.from_label.append(nn.Linear(self.D, self.D).to(self.dtype))
            self.from_type.append(nn.Linear(self.D, self.D).to(self.dtype))
            self.to_node.append(nn.Linear(self.D, self.D).to(self.dtype))
            self.to_type.append(nn.Linear(self.D, self.D).to(self.dtype))
            self.to_feature.append(nn.Linear(self.D, self.D).to(self.dtype))
            self.to_label.append(nn.Linear(self.D, self.D).to(self.dtype))

            if self.attention_scatter:
                self.node_logit.append(nn.Linear(self.D, self.H).to(self.dtype))
                self.feature_logit.append(nn.Linear(self.D, self.H).to(self.dtype))
                self.label_logit.append(nn.Linear(self.D, self.H).to(self.dtype))
                self.type_logit.append(nn.Linear(self.D, self.H).to(self.dtype))

        # pre-norm for additive refinement outer loop + final norm before heads
        if self.additive_refinement:
            if self.layerscale_init > 0:
                _ls = self.layerscale_init
                self.ls_node = nn.Parameter(torch.full((self.D,), _ls, dtype=self.dtype))
                self.ls_type = nn.Parameter(torch.full((self.D,), _ls, dtype=self.dtype))
                self.ls_feature = nn.Parameter(torch.full((self.D,), _ls, dtype=self.dtype))
                self.ls_label = nn.Parameter(torch.full((self.D,), _ls, dtype=self.dtype))
            self.final_norm = RMSNorm(self.D).to(self.dtype)
            if self.use_pre_norm:
                _n_prenorm = self.T if not self.parameter_tying_across_refinements else 1
                self.pre_norm_node = nn.ModuleList(
                    [RMSNorm(self.D).to(self.dtype) for _ in range(_n_prenorm)]
                )
                self.pre_norm_type = nn.ModuleList(
                    [RMSNorm(self.D).to(self.dtype) for _ in range(_n_prenorm)]
                )
                self.pre_norm_feature = nn.ModuleList(
                    [RMSNorm(self.D).to(self.dtype) for _ in range(_n_prenorm)]
                )
                self.pre_norm_label = nn.ModuleList(
                    [RMSNorm(self.D).to(self.dtype) for _ in range(_n_prenorm)]
                )

        # intra-node self-attention
        if self.use_intra_node_attention:
            self.intra_attn = nn.ModuleList()
            for _ in range(self.T if not self.parameter_tying_across_refinements else 1):
                self.intra_attn.append(
                    IntraNodeAttention(
                        dim=self.D,
                        n_heads=self.intra_node_attention_n_heads,
                        multiple_of=self.D,
                        norm_eps=1e-5,
                        use_rotary_embedding=self.rotary_emb_intra_node_attention,
                        use_ffn=self.intra_node_ffn,
                    ).to(self.dtype)
                )

        # learnable gates for shared walk embedding injection (per channel)
        _walk_gate_init = 0.1
        self.walk_emb_gate_node = nn.Parameter(torch.full((self.D,), _walk_gate_init, dtype=self.dtype))
        self.walk_emb_gate_feature = nn.Parameter(torch.full((self.D,), _walk_gate_init, dtype=self.dtype))
        self.walk_emb_gate_label = nn.Parameter(torch.full((self.D,), _walk_gate_init, dtype=self.dtype))

        self._profile_ms = {"walks": 0.0, "intra": 0.0, "global": 0.0, "rw": 0.0}
        self._profile_n = 0

        # link prediction head
        mlp_hidden_dim = 128
        self.head = nn.Sequential(
            nn.Linear(self.D, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, 1),
        ).to(self.dtype)

        # node classification head (one logit per one-hot class channel)
        self.node_cls_head_one_hot = nn.Sequential(
            nn.Linear(self.D, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, 1),
        ).to(self.dtype)
        set_kernel_compile((not self.training) and self.compile_eval)

    def train(self, mode: bool = True):
        super().train(mode)
        set_kernel_compile((not mode) and self.compile_eval)
        return self

    def _make_global_refine_net(
        self, n_channels: int, n_heads: int, model_cfg: dict,
    ) -> nn.ModuleList:
        """One refinement's per-channel SequenceAttention stack for context attention."""
        nets = nn.ModuleList()
        for _ in range(n_channels):
            nets.append(
                SequenceAttention(
                    dim=self.D,
                    n_layers=model_cfg["n_layers"],
                    n_heads=n_heads,
                    multiple_of=self.D,
                    norm_eps=1e-5,
                    use_rotary_embedding=False,
                ).to(self.dtype)
            )
        return nets

    def _net_global_for_task(self, task: str) -> nn.ModuleList:
        """Select context-attention nets for ``\"node\"`` or ``\"link\"`` prediction."""
        if task == "link" and self.untie_attention_across_tasks:
            return self.net_global_link
        return self.net_global

    def _apply_global_channel_attn(
        self,
        net: nn.Module,
        h_chan: Tensor,
        bsize: int,
        num_nodes: int,
        kv_gather_idx: Optional[Tensor],
        kv_key_padding_mask: Optional[Tensor],
        external_kv: Optional[Tuple[Tensor, Tensor]],
    ) -> Tensor:
        """Run global ICL over feature/label channels, chunked on B*C.

        ``h_chan`` is ``[B, N, C, D]``. Each channel is an independent sequence
        of length N; running all C at once with math SDPA is C times one
        ``[H, N, M]`` attention matrix.
        """
        C = int(h_chan.shape[2])
        xc = h_chan.permute(0, 2, 1, 3).reshape(bsize * C, num_nodes, self.D)
        n_seq = bsize * C
        if external_kv is not None:
            k_all, v_all = external_kv
            kv_all_idx = None
            kv_all_pad = None
        elif kv_gather_idx is not None:
            k_all = v_all = None
            kv_all_idx = kv_gather_idx.repeat_interleave(C, dim=0)
            kv_all_pad = (
                kv_key_padding_mask.repeat_interleave(C, dim=0)
                if kv_key_padding_mask is not None
                else None
            )
        else:
            k_all = v_all = None
            kv_all_idx = None
            kv_all_pad = None

        chunk = max(1, int(self.global_channel_chunksize))
        use_ckpt = self.checkpoint_sublayers and torch.is_grad_enabled()
        outs = []
        for i0 in range(0, n_seq, chunk):
            i1 = min(i0 + chunk, n_seq)
            x_chunk = xc[i0:i1].contiguous()
            if k_all is not None:
                ext_i = (k_all[i0:i1], v_all[i0:i1])
                kv_i = None
                pad_i = None
            else:
                ext_i = None
                kv_i = kv_all_idx[i0:i1] if kv_all_idx is not None else None
                pad_i = kv_all_pad[i0:i1] if kv_all_pad is not None else None

            def _run(
                _x,
                _kv=kv_i,
                _pad=pad_i,
                _ext=ext_i,
            ):
                return net(
                    _x,
                    kv_gather_idx=_kv,
                    kv_key_padding_mask=_pad,
                    external_kv=_ext,
                )

            if use_ckpt and ext_i is None:
                y = torch_checkpoint(_run, x_chunk, use_reentrant=False)
            else:
                y = _run(x_chunk)
            outs.append(y)
        xc = torch.cat(outs, dim=0)
        return xc.view(bsize, C, num_nodes, self.D).permute(0, 2, 1, 3)

    def _lp_global_kv_mask_from_roles(
        self, node_roles: Tensor, num_nodes: int,
    ) -> Tensor:
        """Build the LP global-attention K/V mask from roles without changing them.

        Keeps the query node (role 2) and thinking rows. Role-1 query-relation
        neighbors are kept in full unless ``global_kv_neighbors`` is set, in
        which case at most that many are sampled per row.
        """
        bsize, num_graph_nodes = node_roles.shape
        device = node_roles.device
        query = node_roles == 2
        nbr = node_roles == 1
        cap = self.global_kv_neighbors
        if cap is None:
            keep = query | nbr
        else:
            k = min(int(cap), num_graph_nodes)
            scores = torch.rand(
                bsize, num_graph_nodes, device=device, dtype=torch.float32,
            )
            scores = scores.masked_fill(~nbr, -1.0)
            _, idx = scores.topk(k, dim=1)
            keep_nbr = torch.zeros_like(nbr)
            keep_nbr.scatter_(1, idx, True)
            keep_nbr &= nbr
            keep = query | keep_nbr
        mask = torch.zeros(bsize, num_nodes, dtype=torch.bool, device=device)
        mask[:, :num_graph_nodes] = keep
        if self.thinking_rows > 0:
            mask[:, num_graph_nodes:num_graph_nodes + self.thinking_rows] = True
        return mask

    def _profile_enabled(self) -> bool:
        return (not self.training) and os.environ.get("WANDER_PROFILE_EVAL", "0") == "1"

    def _profile_begin(self, key: str, *, cuda: bool = False):
        if not self._profile_enabled() or key not in self._profile_ms:
            return None
        if cuda and torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            return ("cuda", key, start, end)
        return ("cpu", key, time.perf_counter(), None)

    def _profile_end(self, token) -> None:
        if token is None:
            return
        kind, key, start, end = token
        if kind == "cuda":
            end.record()
            end.synchronize()
            self._profile_ms[key] += float(start.elapsed_time(end))
        else:
            self._profile_ms[key] += (time.perf_counter() - start) * 1000.0

    @contextmanager
    def _profile_span(self, key: str, *, cuda: bool = False):
        """Accumulate wall time for one named eval phase (ms)."""
        token = self._profile_begin(key, cuda=cuda)
        try:
            yield
        finally:
            self._profile_end(token)

    def _profile_mark_forward(self) -> None:
        if not self._profile_enabled():
            return
        self._profile_n += 1
        n = self._profile_n
        if n == 1 or n == 5 or n % 50 == 0:
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(self._profile_summary_text(), flush=True)

    def _profile_summary_text(self) -> str:
        n = max(self._profile_n, 1)
        parts = [
            f"{key}={self._profile_ms[key] / n:.1f}ms"
            for key in ("walks", "intra", "global", "rw")
        ]
        total = sum(self._profile_ms[k] for k in ("walks", "intra", "global", "rw")) / n
        return (
            f"[wander profile] n={self._profile_n} per_fwd "
            + " ".join(parts)
            + f" total={total:.1f}ms"
        )

    @staticmethod
    def _all_graph_node_indices(data: Data, device: torch.device) -> Tensor:
        """Return ``0 .. N-1`` for the full graph (aligned with ``data.num_nodes`` / features)."""
        if getattr(data, "num_nodes", None) is not None:
            n = int(data.num_nodes)
        elif getattr(data, "x", None) is not None:
            n = int(data.x.size(0))
        else:
            n = int(data.edge_index.max().item()) + 1 if data.edge_index.numel() > 0 else 0
        return torch.arange(n, device=device, dtype=torch.long)

    def _prune_graph_to_indices(self, data: Data, keep_old: Tensor) -> Tuple[Data, Tensor]:
        """Induced subgraph on keep_old (global ids). Does not mutate ``data``."""
        num_nodes_full = data.num_nodes
        device = keep_old.device
        keep_old = torch.unique(keep_old)
        mapper = torch.full((num_nodes_full,), -1, dtype=torch.long, device=device)
        mapper[keep_old] = torch.arange(keep_old.numel(), device=device, dtype=torch.long)

        new_ei, new_et = subgraph(
            keep_old,
            data.edge_index,
            data.edge_type,
            relabel_nodes=True,
            num_nodes=num_nodes_full,
        )
        out = Data(edge_index=new_ei, edge_type=new_et, num_nodes=int(keep_old.numel()))
        out.num_relations = data.num_relations
        flag = getattr(data, "supervise_non_train_nodes", None)
        if flag is not None:
            out.supervise_non_train_nodes = bool(flag)

        if getattr(data, "x", None) is not None:
            out.x = data.x[keep_old]
        if getattr(data, "y", None) is not None:
            out.y = data.y[keep_old]
        for key in _NODE_ALIGNED_OPTIONAL_KEYS:
            v = getattr(data, key, None)
            if isinstance(v, Tensor) and v.shape[0] == num_nodes_full:
                setattr(out, key, v[keep_old])

        assigned = {
            "edge_index", "edge_type", "num_nodes", "num_relations", "x", "y",
            "train_mask", "val_mask", "test_mask",
        }
        try:
            _keys = getattr(data, "keys", None)
            if _keys is None:
                key_iter = ()
            elif callable(_keys):
                key_iter = _keys()
            else:
                key_iter = list(_keys)
            for key in key_iter:
                if key in assigned:
                    continue
                value = getattr(data, key, None)
                if isinstance(value, Tensor) and value.dim() >= 1 and value.shape[0] == num_nodes_full:
                    setattr(out, key, value[keep_old])
        except (TypeError, AttributeError, KeyError):
            pass
        return out, mapper

    def _remap_walk_tuple(self, walk_tuple: tuple, mapper: Tensor) -> tuple:
        walks, named_walks, restarts, neighbors, types, named_types, directions = walk_tuple
        walks = mapper[walks]
        if self.record_neighbors:
            neighbors = mapper[neighbors]
        return walks, named_walks, restarts, neighbors, types, named_types, directions

    original_graph_visit_stats = staticmethod(_original_graph_visit_stats)

    def _set_eval_coverage_stats(
        self,
        coverage_after_first: Optional[Dict[str, float]],
        rw_node_mask: Tensor,
        rw_type_mask: Tensor,
        num_types: int,
        graph_visit_stats: Optional[Dict[str, float]],
    ) -> None:
        if self.training or coverage_after_first is None:
            return
        self._coverage_stats = {
            "node_pct_1step": coverage_after_first["node"],
            "type_pct_1step": coverage_after_first["type"],
            "node_pct_total": (rw_node_mask.float().mean(dim=1).mean() * 100).item(),
            "type_pct_total": (rw_type_mask[:, :num_types].float().mean(dim=1).mean() * 100).item(),
        }
        if graph_visit_stats:
            self._coverage_stats.update(graph_visit_stats)

    def _num_features_for_walk_cap(self, data: Data) -> int:
        """Feature count used for walk_num memory cap (matches forward grouping)."""
        x = getattr(data, "x", None)
        if x is None or x.ndim != 2 or x.shape[1] == 0:
            return self.thinking_features
        n = int(x.shape[1])
        if self.feature_groups > 1:
            n = math.ceil(n / self.feature_groups)
        return n + self.thinking_features

    def _effective_walk_params(
        self,
        num_nodes: int,
        num_features: int,
        base_n: Optional[int] = None,
        start_mode: str = "default",
    ) -> Tuple[int, int]:
        if base_n is None:
            base_n = self._walk_num_for_start_mode(start_mode)
        n_cfg = int(base_n)
        if self.adaptive_walks:
            n, l = n_cfg, self.L_max
            if num_nodes < 1000:
                n, l = n // 4, self.L_max // 4
            elif num_nodes < 8000:
                n, l = n // 2, self.L_max // 2
        else:
            n, l = n_cfg, self.L
        # Cap walk count only during training (memory); eval / analysis use full budget.
        if getattr(self, "training", False) and num_features > _FEATURE_WALK_NUM_CAP_THRESHOLD:
            n = min(n, _FEATURE_WALK_NUM_CAP)
        return max(1, n), max(1, l)

    def _walk_num_for_start_mode(self, start_mode: str) -> int:
        """Base walk_num before adaptive scaling / feature cap.

        Cached train-KV Phase A (``train_chunk`` / ``train_only``) can use
        ``train_kv_walk_num`` independently of training / eval walk budgets.
        Eval forwards (``not self.training``) can use ``eval_walk_num`` when set.
        """
        if start_mode in ("train_chunk", "train_only"):
            n = getattr(self, "train_kv_walk_num", None)
            if n is not None:
                return int(n)
        if not self.training:
            eval_n = getattr(self, "eval_walk_num", None)
            if eval_n is not None:
                return int(eval_n)
        return self.N

    def _walk(
        self,
        data: Data,
        prefix: Tensor,
        remove_loops: bool,
        prefix_types: Optional[Tensor] = None,
        walk_len: Optional[int] = None,
        stay_on_cpu: bool = False,
    ) -> Tuple:
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        num_types = int(data.num_relations)
        device = prefix.device

        assert isinstance(edge_index, Tensor)
        assert isinstance(num_nodes, int)
        assert isinstance(device, torch.device)

        eff_L = self.L if walk_len is None else walk_len
        return self._walk_impl(
            data,
            prefix,
            remove_loops,
            prefix_types,
            eff_L,
            edge_index,
            num_nodes,
            num_types,
            device,
            stay_on_cpu=stay_on_cpu,
        )

    def warmup_walk_cache(self, data: Data) -> None:
        """Pre-build typed and RW CSRs for both ``remove_loops`` variants (NC + LP)."""
        if not self.use_walk_cache:
            return
        cache = get_walk_graph_cache(data, self)
        get_rw_profile(cache, data, self, remove_loops=True)
        get_rw_profile(cache, data, self, remove_loops=False)

    def _walk_impl(
        self,
        data: Data,
        prefix: Tensor,
        remove_loops: bool,
        prefix_types: Optional[Tensor],
        eff_L: int,
        edge_index: Tensor,
        num_nodes: int,
        num_types: int,
        device: torch.device,
        stay_on_cpu: bool = False,
    ) -> Tuple:
        if not self.use_walk_cache:
            return self._walk_impl_uncached(
                data,
                prefix,
                remove_loops,
                prefix_types,
                eff_L,
                edge_index,
                num_nodes,
                num_types,
                device,
                stay_on_cpu=stay_on_cpu,
            )

        cache = get_walk_graph_cache(data, self)
        rw_profile = get_rw_profile(
            cache,
            data,
            self,
            remove_loops,
        )
        forward_csr = cache.typed_forward
        transposed_csr = cache.typed_transposed

        walks, restarts = graph_walker.random_walks_with_precomputed_probs(
            rw_profile.rw_indptr,
            rw_profile.rw_indices,
            rw_profile.rw_probs,
            n_walks=1,
            walk_len=eff_L,
            p=1,
            q=1,
            alpha=0.0,
            k=None,
            no_backtrack=True,
            prefix=prefix.cpu(),
            verbose=False,
            fast_uniform=self.fast_uniform_walks,
        )

        homogeneous = num_types <= 1
        parse_seed = graph_walker._seed(None)

        if not self.record_neighbors:
            named_walks = graph_walker._anonymize(walks)
            neighbors = np.zeros_like(restarts, dtype=np.int32)

            types, directions = graph_walker._parse_edge_types_and_directions(
                walks,
                restarts,
                *forward_csr,
                *transposed_csr,
                parse_seed,
                homogeneous,
            )

        else:
            assert rw_profile.neighbor_indptr is not None
            assert rw_profile.neighbor_indices is not None
            named_walks, walks, restarts, neighbors = (
                graph_walker._anonymize_with_neighbors(
                    walks,
                    restarts,
                    rw_profile.neighbor_indptr,
                    rw_profile.neighbor_indices,
                )
            )

            types, directions = (
                graph_walker._parse_edge_types_and_directions_with_neighbors(
                    walks,
                    restarts,
                    neighbors,
                    *forward_csr,
                    *transposed_csr,
                    parse_seed,
                    homogeneous,
                )
            )

        if prefix_types is not None:
            assert prefix.ndim == 2 and prefix.shape[1] == 2
            types[:, 1] = np.asarray(prefix_types.cpu(), dtype=np.int32)
            directions[:, 1] = 0

        named_types = graph_walker._anonymize_edge_types(types)

        if stay_on_cpu:
            return self._finalize_cpu_walk_tuple(
                (walks, named_walks, restarts, neighbors, types, named_types, directions),
                num_types,
            )

        return self._walk_tuple_to_device(
            (walks, named_walks, restarts, neighbors, types, named_types, directions),
            device,
            num_types,
        )

    def _walk_impl_uncached(
        self,
        data: Data,
        prefix: Tensor,
        remove_loops: bool,
        prefix_types: Optional[Tensor],
        eff_L: int,
        edge_index: Tensor,
        num_nodes: int,
        num_types: int,
        device: torch.device,
        stay_on_cpu: bool = False,
    ) -> Tuple:
        # for random walks, treat edges as undirected, untyped and remove duplicates
        # remove loops if specified
        edge_index_ = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
        edge_index_ = torch.unique(edge_index_, dim=1)
        if remove_loops:
            edge_index_ = edge_index_[:, edge_index_[0] != edge_index_[1]]

        # CSRs used by the edge-type/direction parser. The forward CSR provides
        # "downstream" (along-walk) candidates; the transposed CSR provides
        # "upstream" candidates, and the parser randomly picks among all matches.
        # Only when inverse edges are added AND only_forward_edges_when_inverses_added
        # is set do we hand the parser an empty transposed CSR: since the graph is
        # then symmetric, every traversed pair still has a forward edge, so each
        # step is labelled with the relation pointing ALONG the walk direction
        # (the "first CSR") instead of randomly mixing in the reverse-relation
        # candidate. Otherwise we keep the original random forward/reverse pick.
        forward_csr = to_csr_tensor(edge_index, data.edge_type, num_nodes)
        if self.add_inverse_edges_kgs and self.only_forward_edges_when_inverses_added:
            empty_edges = edge_index.new_zeros((2, 0))
            empty_types = data.edge_type.new_zeros((0,))
            transposed_csr = to_csr_tensor(empty_edges, empty_types, num_nodes)
        else:
            transposed_csr = to_csr_tensor(edge_index[[1, 0]], data.edge_type, num_nodes)

        # run random walks
        walks, restarts = graph_walker.random_walks_fast(
            graph=Data(
                edge_index=edge_index_, num_nodes=num_nodes, is_directed_hash=False
            ),
            n_walks=1,
            walk_len=eff_L,
            p=1,
            q=1,
            alpha=0.0,
            k=None,
            no_backtrack=True,
            prefix=prefix.cpu(),
            verbose=False,
            fast_uniform=self.fast_uniform_walks,
        )

        homogeneous = num_types <= 1
        parse_seed = graph_walker._seed(None)

        if not self.record_neighbors:
            # anonymize node
            named_walks = graph_walker._anonymize(walks)
            neighbors = np.zeros_like(restarts, dtype=np.int32)

            # parse edge types and directions
            # for this, use the original edge_index
            types, directions = graph_walker._parse_edge_types_and_directions(
                walks,
                restarts,
                *forward_csr,
                *transposed_csr,
                parse_seed,
                homogeneous,
            )

        else:
            # anonymize node and record named neighbors
            named_walks, walks, restarts, neighbors = (
                graph_walker._anonymize_with_neighbors(
                    walks,
                    restarts,
                    *to_csr_tensor(
                        edge_index_, torch.ones_like(edge_index_[0]), num_nodes
                    )[:2],
                )
            )

            # parse edge types and directions
            # for this, use the original edge_index
            types, directions = (
                graph_walker._parse_edge_types_and_directions_with_neighbors(
                    walks,
                    restarts,
                    neighbors,
                    *forward_csr,
                    *transposed_csr,
                    parse_seed,
                    homogeneous,
                )
            )

        # fix first-step edge type (prefix) and direction (0; downstream)
        if prefix_types is not None:
            assert prefix.ndim == 2 and prefix.shape[1] == 2
            types[:, 1] = np.asarray(prefix_types.cpu(), dtype=np.int32)
            directions[:, 1] = 0

        # anonymize edge types, ignoring no-type markers (-1)
        named_types = graph_walker._anonymize_edge_types(types)

        if stay_on_cpu:
            return self._finalize_cpu_walk_tuple(
                (walks, named_walks, restarts, neighbors, types, named_types, directions),
                num_types,
            )

        return self._walk_tuple_to_device(
            (walks, named_walks, restarts, neighbors, types, named_types, directions),
            device,
            num_types,
        )

    def _concat_walk_tensors(self, walk_tuple: tuple) -> tuple:
        """Concatenate all walks per (refinement_step, batch) into one long walk.

        Input tensors have shape [T, bsize, num_walks, L].
        Output tensors have shape [T, bsize, 1, num_walks*L].
        The restart tensor is updated so that the first position of every
        walk except the very first is marked as a restart (=1).
        """
        walks, named_walks, restarts, neighbors, types, named_types, directions = walk_tuple
        restarts = restarts.clone()
        restarts[:, :, 1:, 0] = 1
        out = []
        for t in (walks, named_walks, restarts, neighbors, types, named_types, directions):
            T, B, NW, L = t.shape
            out.append(t.reshape(T, B, 1, NW * L))
        return tuple(out)

    def walks_link_prediction(
        self, data: Data, head_index: Tensor, stay_on_cpu: bool = False,
    ) -> tuple:
        device = head_index.device if not stay_on_cpu else torch.device("cpu")
        num_nodes = data.num_nodes
        num_types = int(data.num_relations)
        bsize = head_index.shape[0]
        sample_T = 1 if self.fixed_rw_across_refinements else self.T
        num_features = self._num_features_for_walk_cap(data)
        eff_N, eff_L = self._effective_walk_params(num_nodes, num_features)

        assert isinstance(data.edge_index, Tensor)
        assert isinstance(num_nodes, int)

        head_on_device = head_index.to(device)
        # random walks starting at: head nodes and random nodes
        # sample starting nodes
        start_nodes = torch.cat(
            [
                head_on_device.repeat(eff_N * sample_T),
                torch.randint(num_nodes, (eff_N * sample_T * bsize,), device=device),
            ],
            dim=0,
        )
        assert start_nodes.numel() == 2 * eff_N * sample_T * bsize
        # random walks
        walks, named_walks, restarts, neighbors, types, named_types, directions = (
            self._walk(
                data, start_nodes, remove_loops=True, walk_len=eff_L,
                stay_on_cpu=stay_on_cpu,
            )
        )

        # random walks starting at: edge types
        # number of walks per edge type, iteration and batch instance
        n1 = (eff_N // num_types) + 1
        # number of walks per edge type
        n2 = n1 * sample_T * bsize
        # sample starting edges
        start_edges, start_types = balanced_sample(
            data.edge_index, data.edge_type, num_types, n2
        )
        # random walks
        (
            walks_,
            named_walks_,
            restarts_,
            neighbors_,
            types_,
            named_types_,
            directions_,
        ) = self._walk(
            data,
            start_edges,
            remove_loops=False,
            prefix_types=start_types,
            walk_len=eff_L,
            stay_on_cpu=stay_on_cpu,
        )

        # number of actually sampled walks, per iteration and batch instance
        # we generally expect _N >= eff_N, but the opposite may happen (e.g., NELLInductive:v4)
        _N = start_edges.shape[0] // (sample_T * bsize)

        # sample permutations for subsampling
        # same across attributes (walks, restarts, ...) and walk length
        # but different for each iteration and batch instance
        perms = torch.argsort(torch.rand(sample_T * bsize, _N, device=device))
        perms = perms.view(sample_T, bsize, _N).permute(2, 0, 1)
        perms = perms[..., None].expand(_N, sample_T, bsize, eff_L)

        def _walk_cpu_tensor(x: Tensor | np.ndarray) -> Tensor:
            if isinstance(x, Tensor):
                return x.to(device=device, dtype=torch.long)
            return torch.tensor(x.astype(np.int32), dtype=torch.long, device=device)

        # combine walks
        def combine(x: Tensor | np.ndarray, y: Tensor | np.ndarray) -> Tensor:
            x = _walk_cpu_tensor(x)
            y = _walk_cpu_tensor(y)
            assert x.ndim == y.ndim == 2
            x = x.view(2 * eff_N, sample_T, bsize, eff_L)
            y = y.view(_N, sample_T, bsize, eff_L)
            # subsampling
            y = y.gather(0, perms)
            if _N >= eff_N:
                y = y[: eff_N]
            else:
                y = y.repeat(eff_N // _N + 1, 1, 1, 1)[: eff_N]
            out = torch.cat([x, y], dim=0).permute(1, 2, 0, 3)
            if self.fixed_rw_across_refinements:
                out = out.repeat(self.T, 1, 1, 1)
            return out

        result = (
            combine(walks, walks_),
            combine(named_walks, named_walks_),
            combine(restarts, restarts_),
            combine(neighbors, neighbors_),
            combine(types, types_),
            combine(named_types, named_types_),
            combine(directions, directions_),
        )
        if self.concat_walks:
            result = self._concat_walk_tensors(result)
        if stay_on_cpu:
            return self._finalize_cpu_walk_tuple(result, num_types)
        return result

    def _sample_nc_train_starts(
        self, data: Data, n_starts: int, num_groups: int, walk_device: torch.device,
    ) -> Tensor:
        """[n_starts, num_groups] uniform train-node walk starts."""
        num_nodes = data.num_nodes
        train_mask = getattr(data, "train_mask", None)
        if (
            train_mask is not None
            and train_mask.numel() == num_nodes
            and int(train_mask.sum().item()) > 0
        ):
            train_idx = train_mask.nonzero(as_tuple=True)[0].to(device=walk_device)
            return _uniform_train_starts(
                train_idx, n_starts, num_groups, walk_device,
            )
        return torch.randint(num_nodes, (n_starts, num_groups), device=walk_device)

    def _reshape_nc_walks(
        self, walk_tuple: tuple, data: Data, walks_per_sample: int, sample_T: int,
        num_groups: int, eff_L: int, walk_device: torch.device, stay_on_cpu: bool,
    ) -> tuple:
        """Reshape raw walk channels to [T, num_groups, walks_per_sample, L]."""

        def _walk_cpu_tensor(x: Tensor | np.ndarray) -> Tensor:
            if isinstance(x, Tensor):
                return x.to(device=walk_device, dtype=torch.long)
            return torch.tensor(x.astype(np.int32), dtype=torch.long, device=walk_device)

        def reshape(x: Tensor | np.ndarray) -> Tensor:
            x = _walk_cpu_tensor(x)
            out = x.view(walks_per_sample, sample_T, num_groups, eff_L).permute(1, 2, 0, 3)
            if self.fixed_rw_across_refinements:
                out = out.repeat(self.T, 1, 1, 1)
            return out

        result = tuple(reshape(ch) for ch in walk_tuple)
        if self.concat_walks:
            result = self._concat_walk_tensors(result)
        if stay_on_cpu:
            return self._finalize_cpu_walk_tuple(result, int(data.num_relations))
        return result

    def _walk_nc_starts(
        self,
        data: Data,
        start_nodes: Tensor,
        *,
        walk_len: int,
        stay_on_cpu: bool,
    ) -> tuple:
        """NC walks from ``start_nodes``."""
        flat = start_nodes.reshape(-1)
        return self._walk(
            data, flat, remove_loops=True,
            walk_len=walk_len, stay_on_cpu=stay_on_cpu,
        )

    def _walks_nc_single_source(
        self, data: Data, batch_indices: Tensor, start_mode: str,
        eff_N: int, eff_L: int, sample_T: int, num_groups: int,
        walk_device: torch.device, stay_on_cpu: bool,
        walk_focus_indices: Optional[Tensor] = None,
    ) -> tuple:
        """Full-length NC walks from a single start source (train or batch nodes).

        Both ``train_only`` (legacy Phase A) and ``batch_only`` (Phase B cached
        eval) use the full combined walk budget (~3 * eff_N walks per refinement
        step). ``batch_only`` assigns starts via stratified sampling over batch
        nodes. ``train_chunk`` stratifies starts over ``walk_focus_indices``
        without treating those nodes as query/batch nodes in the forward pass.
        """
        n_starts = _nc_combined_walk_start_count(eff_N, sample_T)
        walks_per_sample = max(1, n_starts // sample_T)
        n_starts = walks_per_sample * sample_T

        if start_mode == "train_only":
            starts = self._sample_nc_train_starts(
                data, n_starts, num_groups, walk_device,
            )
        elif start_mode == "train_chunk":
            assert walk_focus_indices is not None and walk_focus_indices.numel() > 0, (
                "train_chunk requires non-empty walk_focus_indices"
            )
            assert num_groups == 1
            starts = _stratified_batch_starts(
                walk_focus_indices.to(walk_device), walks_per_sample, sample_T,
                walk_device,
            ).unsqueeze(1)
        else:
            assert num_groups == 1
            starts = _stratified_batch_starts(
                batch_indices.to(walk_device), walks_per_sample, sample_T,
                walk_device,
            ).unsqueeze(1)

        walk_tuple = self._walk_nc_starts(
            data, starts.flatten(), walk_len=eff_L, stay_on_cpu=stay_on_cpu,
        )

        reshaped = self._reshape_nc_walks(
            walk_tuple, data, walks_per_sample, sample_T, num_groups, eff_L,
            walk_device, stay_on_cpu,
        )
        return self._maybe_filter_train_free_walks(reshaped, data, start_mode)

    def _maybe_filter_train_free_walks(
        self, walk_tuple: tuple, data: Data, start_mode: str,
    ) -> tuple:
        """Drop Phase B walks that never visit a train node (keep_prob=0)."""
        keep_prob = getattr(self, "keep_train_free_p", None)
        if (
            keep_prob is None
            or self.training
            or start_mode != "batch_only"
        ):
            return walk_tuple
        train_mask = getattr(data, "train_mask", None)
        if train_mask is None:
            return walk_tuple
        return _filter_train_free_walks(walk_tuple, train_mask, float(keep_prob))

    def walks_node_classification(
        self, data: Data, batch_indices: Tensor, stay_on_cpu: bool = False,
        start_mode: str = "default",
        walk_focus_indices: Optional[Tensor] = None,
    ) -> tuple:
        """Generate walks for node classification.

        Start modes:
            "default":    random-start walks sample train nodes (uniform or
                          class-balanced); batch-start walks sample query nodes
                          from ``batch_indices`` (current behaviour).
            "train_only": all walks start from train nodes at full walk length
                          (legacy Phase A). Uses the full combined walk budget
                          from ``train_kv_walk_num`` when cached train-KV is on.
            "train_chunk": stratified starts over ``walk_focus_indices`` (chunked
                          Phase A); ``batch_indices`` does not control walks.
                          Walk budget is ``train_kv_walk_num`` (else ``walk_num``).
            "batch_only": all walks start from ``batch_indices``, assigned via
                          vectorized stratified sampling; uses the same ~3*eff_N
                          walk budget per refinement as the default combined path
                          (Phase B of the cached train-KV eval; ``eval_walk_num``
                          when set, else ``walk_num``).

        Args:
            batch_indices: [K] query node indices for this forward pass.
            walk_focus_indices: [K'] nodes for ``train_chunk`` walk starts.

        Returns:
            Tuple of walk tensors, each shaped [T, 1, walks_per_sample, L].
        """
        assert start_mode in ("default", "train_only", "batch_only", "train_chunk")
        device = batch_indices.device
        walk_device = torch.device("cpu") if stay_on_cpu else device
        num_nodes = data.num_nodes
        num_groups = 1
        K = batch_indices.shape[0]
        sample_T = 1 if self.fixed_rw_across_refinements else self.T
        num_features = self._num_features_for_walk_cap(data)
        eff_N, eff_L = self._effective_walk_params(
            num_nodes, num_features, base_n=self._walk_num_for_start_mode(start_mode),
        )
        assert isinstance(data.edge_index, Tensor)
        assert isinstance(num_nodes, int)

        if start_mode != "default":
            return self._walks_nc_single_source(
                data, batch_indices, start_mode, eff_N, eff_L, sample_T,
                num_groups, walk_device, stay_on_cpu,
                walk_focus_indices=walk_focus_indices,
            )

        n_random_starts, n_batch_starts, walks_per_sample = _nc_walk_start_counts(
            eff_N, sample_T, self.short_random_starts,
        )
        use_short = self.short_random_starts
        random_walk_len = max(1, eff_L // 2) if use_short else eff_L

        random_starts = self._sample_nc_train_starts(
            data, n_random_starts, num_groups, walk_device,
        )
        rand_k = torch.randint(K, (n_batch_starts,), device=walk_device)
        batch_starts = batch_indices.to(walk_device)[rand_k].unsqueeze(1)

        def _walk_cpu_tensor(x: Tensor | np.ndarray) -> Tensor:
            if isinstance(x, Tensor):
                return x.to(device=walk_device, dtype=torch.long)
            return torch.tensor(x.astype(np.int32), dtype=torch.long, device=walk_device)

        def reshape(x: Tensor | np.ndarray) -> Tensor:
            x = _walk_cpu_tensor(x)
            out = x.view(walks_per_sample, sample_T, num_groups, eff_L).permute(1, 2, 0, 3)
            if self.fixed_rw_across_refinements:
                out = out.repeat(self.T, 1, 1, 1)
            return out

        if random_walk_len == eff_L:
            all_starts = torch.cat([random_starts, batch_starts], dim=0)
            start_nodes = all_starts.flatten()
            walk_tuple = self._walk_nc_starts(
                data, start_nodes, walk_len=eff_L, stay_on_cpu=stay_on_cpu,
            )
        else:
            random_nodes = random_starts.flatten()
            batch_nodes = batch_starts.flatten()
            random_tuple = self._walk_nc_starts(
                data, random_nodes, walk_len=random_walk_len, stay_on_cpu=stay_on_cpu,
            )
            batch_tuple = self._walk_nc_starts(
                data, batch_nodes, walk_len=eff_L, stay_on_cpu=stay_on_cpu,
            )
            random_tuple = _pad_walk_tuple(random_tuple, eff_L)
            walk_tuple = _concat_walk_tuples(random_tuple, batch_tuple)

        result = tuple(reshape(ch) for ch in walk_tuple)
        if self.concat_walks:
            result = self._concat_walk_tensors(result)
        if stay_on_cpu:
            return self._finalize_cpu_walk_tuple(result, int(data.num_relations))
        return result

    def _process_attr_channel(
        self,
        h_attr: Tensor,
        x: Tensor,
        _walks: Tensor,
        walks_t: Tensor,
        net: nn.Module,
        from_proj: nn.Module,
        to_proj: nn.Module,
        logit_proj: Optional[nn.Module],
        num_nodes: int,
        walk_emb_gate: Optional[nn.Parameter] = None,
        return_seq: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Process a feature or label channel through gather-project-GRU-scatter,
        chunking over the attribute dimension K to bound peak memory.

        Args:
            h_attr:     [bsize, num_nodes, K, D] per-node attribute states
            x:          [bsize, samples, L, D]   shared walk embedding
            _walks:     [bsize, samples*L]        flattened walk node indices
            walks_t:    [bsize, samples, L]        walk node indices for this refinement
            net:        BidirectionalGRU module
            from_proj:  input projection (e.g. self.from_feature[i])
            to_proj:    output projection for scatter (e.g. self.to_feature[i])
            logit_proj: logit projection for attention scatter, or None
            num_nodes:  number of graph nodes
            return_seq: if True, also return the per-step processed walk reps
                        ``[bsize, K*samples, walk_len, D]`` (channel-major over K,
                        then samples) for the joint relation scatter. Forces the
                        per-net checkpoint path (the combined-chunk checkpoint is
                        skipped since the sequence must be materialized anyway).

        Returns:
            Tuple of:
            - [bsize, num_nodes, K, D] scattered attribute updates
            - Optional [bsize, K*samples, walk_len, D] per-step reps (return_seq)
        """
        bsize, _, K, D = h_attr.shape
        samples = walks_t.shape[-2]
        walk_len = walks_t.shape[-1]
        cs = self.inter_node_chunksize or K
        # Bidirectional GRU autograd is ~1GB/channel at samples=walk_len=128.
        # FULL_CORA K=70 as one chunk is ~70GB on the backward recompute.
        max_gru_batch = 1024
        gru_batch = int(bsize) * min(int(cs), int(K)) * int(samples)
        if gru_batch > max_gru_batch:
            cs = max(1, max_gru_batch // max(int(samples), 1))

        parts = []
        seq_parts: list = []
        use_full_attr_chunk_ckpt = (
            self.checkpoint_sublayers
            and torch.is_grad_enabled()
        )

        def attr_chunk_forward(
            h_attr_k: Tensor, x_: Tensor, _walks_: Tensor, walks_t_: Tensor,
        ) -> Tuple[Tensor, Tensor]:
            b, _, kc_, d = h_attr_k.shape
            xc_ = h_attr_k.gather(1, _walks_[:, :, None, None].expand(-1, -1, kc_, d))
            xc_ = xc_.view(b, samples, walk_len, kc_, d).permute(0, 3, 1, 2, 4)
            xc_ch = from_proj(xc_)
            x_gate = (walk_emb_gate * x_).unsqueeze(1) if walk_emb_gate is not None else x_.unsqueeze(1)
            xc_ = xc_ch + x_gate
            xc_ = xc_.reshape(b * kc_ * samples, walk_len, d)
            xc_ = net(xc_)
            wc_ = walks_t_.unsqueeze(1).expand(-1, kc_, -1, -1).reshape(b * kc_, samples, walk_len)
            xc_proj = to_proj(xc_)
            if logit_proj is not None:
                xc_logits = logit_proj(xc_)
            else:
                xc_logits = None
            if xc_logits is not None:
                sc_ = consensus_softmax_stable(xc_proj, xc_logits, wc_, num_nodes)
            else:
                sc_ = consensus_mean(xc_proj, wc_, num_nodes)
            sc_out = sc_.view(b, kc_, num_nodes, d).permute(0, 2, 1, 3)
            seq_out = xc_.view(b, kc_, samples, walk_len, d)
            return sc_out, seq_out

        def attr_chunk_forward_sc(
            h_attr_k: Tensor, x_: Tensor, _walks_: Tensor, walks_t_: Tensor,
        ) -> Tensor:
            sc_out, _seq = attr_chunk_forward(h_attr_k, x_, _walks_, walks_t_)
            return sc_out

        if os.environ.get("LOG_WANDER_MEM", "").strip():
            alloc = (
                torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            )
            print(
                f"[wander_rw] K={K} cs={cs} samples={samples} walk_len={walk_len} "
                f"return_seq={return_seq} ckpt={use_full_attr_chunk_ckpt} "
                f"alloc_gb={alloc:.2f}",
                flush=True,
            )

        for k0 in range(0, K, cs):
            k1 = min(k0 + cs, K)
            kc = k1 - k0
            h_attr_k = h_attr[:, :, k0:k1, :]

            if use_full_attr_chunk_ckpt:
                if return_seq:
                    sc_out, seq_out = torch_checkpoint(
                        attr_chunk_forward,
                        h_attr_k,
                        x,
                        _walks,
                        walks_t,
                        use_reentrant=False,
                    )
                    parts.append(sc_out)
                    seq_parts.append(seq_out)
                else:
                    parts.append(
                        torch_checkpoint(
                            attr_chunk_forward_sc,
                            h_attr_k,
                            x,
                            _walks,
                            walks_t,
                            use_reentrant=False,
                        )
                    )
                continue

            xc = h_attr_k.gather(1, _walks[:, :, None, None].expand(-1, -1, kc, D))
            xc = xc.view(bsize, samples, walk_len, kc, D)
            xc = xc.permute(0, 3, 1, 2, 4)
            xc_channel = from_proj(xc)
            x_gated = (walk_emb_gate * x).unsqueeze(1) if walk_emb_gate is not None else x.unsqueeze(1)
            xc = xc_channel + x_gated
            xc = xc.reshape(bsize * kc * samples, walk_len, D)
            if self.checkpoint_sublayers and torch.is_grad_enabled():
                xc = torch_checkpoint(net, xc, use_reentrant=False)
            else:
                xc = net(xc)

            if return_seq:
                seq_parts.append(xc.view(bsize, kc, samples, walk_len, D))

            wc = walks_t.unsqueeze(1).expand(-1, kc, -1, -1)
            wc = wc.reshape(bsize * kc, samples, walk_len)
            xc_proj = to_proj(xc)
            xc_logits = logit_proj(xc) if logit_proj is not None else None
            if xc_logits is not None:
                sc = consensus_softmax_stable(xc_proj, xc_logits, wc, num_nodes)
            else:
                sc = consensus_mean(xc_proj, wc, num_nodes)
            sc = sc.view(bsize, kc, num_nodes, D).permute(0, 2, 1, 3)
            parts.append(sc)

        out = torch.cat(parts, dim=2) if len(parts) > 1 else parts[0]
        seq = None
        if return_seq:
            # [bsize, K, samples, walk_len, D] -> [bsize, K*samples, walk_len, D]
            seq = torch.cat(seq_parts, dim=1) if len(seq_parts) > 1 else seq_parts[0]
            seq = seq.reshape(bsize, K * samples, walk_len, D)
        return out, seq

    def _refinement_step(
        self, t, h_node, h_feature, h_label, h_type,
        rw_node_mask, rw_type_mask,
        walks_t, named_walks_t, restarts_t, neighbors_t,
        types_t, named_types_t, directions_t,
        samples, num_nodes, num_types,
        has_features, has_labels, bsize,
        head_index=None, type_index=None,
        global_kv_mask: Optional[Tensor] = None,
        external_global_kv: Optional[Dict[str, Tuple[Tensor, Tensor]]] = None,
        capture_train_rows: Optional[Tensor] = None,
        capture_out: Optional[list] = None,
        task: str = "node",
    ):
        """One refinement step (intra-node attn -> global attn -> RW update).

        Cached train-KV eval extensions:
            external_global_kv: per-channel pre-projected (k, v) used as K/V in
                the global-attention nets instead of ``global_kv_mask`` gathers
                (Phase B). Keys: "node" ([1, M, D] pair), "feature"
                ([F, M, D] pair), "label" ([C, M, D] pair).
            capture_train_rows / capture_out: when set (Phase A), the hidden
                states at the given local rows are snapshotted right before the
                global-attention block and appended to ``capture_out``.
            task: ``\"node\"`` or ``\"link\"`` — selects which ``net_global`` to use
                when ``untie_attention_across_tasks`` is enabled.
        """
        device = h_node.device
        net_global = self._net_global_for_task(task)
        profile_cuda = bool(getattr(h_node, "is_cuda", False))
        h_node, h_feature, h_label = self._materialize_token_states(
            h_node, h_feature, h_label, has_features, has_labels,
        )

        # --- Phase 1: intra-node self-attention (chunked over B*N) ---
        _prof_intra = self._profile_begin("intra", cuda=profile_cuda)
        if self.use_intra_node_attention:
            t_intra = 0 if self.parameter_tying_across_refinements else t
            intra_mod = self.intra_attn[t_intra]

            num_features = h_feature.shape[2] if has_features else 0
            num_label_tokens = h_label.shape[2] if has_labels else 0

            h_node_flat = h_node.reshape(bsize * num_nodes, self.D)
            h_node_new = torch.empty_like(h_node_flat)
            if has_features:
                h_feat_flat = h_feature.reshape(bsize * num_nodes, num_features, self.D)
                h_feat_new = torch.empty_like(h_feat_flat)
            if has_labels:
                h_lab_flat = h_label.reshape(bsize * num_nodes, num_label_tokens, self.D)
                h_lab_new = torch.empty_like(h_lab_flat)

            BN = bsize * num_nodes
            cs = self.intra_node_chunksize # we chunk over the node(*batch) dimension
            use_ckpt = self.checkpoint_sublayers and torch.is_grad_enabled()

            for i0 in range(0, BN, cs):
                i1 = min(i0 + cs, BN)
                tokens = [h_node_flat[i0:i1, None, :]]
                if has_features:
                    tokens.append(h_feat_flat[i0:i1])
                if has_labels:
                    tokens.append(h_lab_flat[i0:i1])
                x_chunk = torch.cat(tokens, dim=1)

                if use_ckpt:
                    x_chunk = torch_checkpoint(
                        intra_mod.forward_tokens,
                        x_chunk,
                        None,
                        use_reentrant=False,
                    )
                else:
                    x_chunk = intra_mod.forward_tokens(
                        x_chunk,
                        attn_mask=None,
                    )

                h_node_new[i0:i1] = x_chunk[:, 0, :]
                if has_features:
                    h_feat_new[i0:i1] = x_chunk[:, 1:1 + num_features, :]
                if has_labels:
                    h_lab_new[i0:i1] = x_chunk[:, 1 + num_features:, :]

            h_node = h_node_new.reshape(bsize, num_nodes, self.D)
            if has_features:
                h_feature = h_feat_new.reshape(bsize, num_nodes, num_features, self.D)
            if has_labels:
                h_label = h_lab_new.reshape(bsize, num_nodes, num_label_tokens, self.D)
        self._profile_end(_prof_intra)

        # --- Phase 2: in context learning (attention to context nodes) ---
        t_inter = 0 if self.parameter_tying_across_refinements else t

        # Snapshot pre-global-attention hidden states at train rows (Phase A of
        # the cached train-KV eval). Captured post intra-node attention, i.e.
        # exactly the states the global-attention K/V would be projected from.
        if capture_out is not None and capture_train_rows is not None:
            snapshot = {
                "node": h_node[0, capture_train_rows].detach(),
            }
            if has_features:
                snapshot["feature"] = h_feature[0, capture_train_rows].detach()
            if has_labels:
                snapshot["label"] = h_label[0, capture_train_rows].detach()
            capture_out.append(snapshot)

        _prof_global = self._profile_begin("global", cuda=profile_cuda)
        if self.global_attention_update:
            # Build padded gather indices once per refinement step when the
            # caller restricts K/V to a per-row subset of nodes.
            if external_global_kv is not None:
                assert global_kv_mask is None, (
                    "external_global_kv is mutually exclusive with global_kv_mask"
                )
                kv_gather_idx = None
                kv_key_padding_mask = None
            elif global_kv_mask is not None:
                counts = global_kv_mask.sum(dim=1)
                assert int(counts.min().item()) > 0, (
                    "global_kv_mask must be non-empty per row"
                )
                max_m = int(counts.max().item())
                min_m = int(counts.min().item())
                sorted_idx = torch.argsort(~global_kv_mask, dim=1, stable=True)
                kv_gather_idx = sorted_idx[:, :max_m]
                # bsize=1 always has min_m==max_m; a dense True padding mask
                # still forces math SDPA (~C × 27GB on FULL_CORA labels).
                kv_key_padding_mask = (
                    None
                    if min_m == max_m
                    else (torch.arange(max_m, device=device)[None] < counts[:, None])
                )
            else:
                kv_gather_idx = None
                kv_key_padding_mask = None

            # --- node channel ---
            if self.checkpoint_sublayers and torch.is_grad_enabled():
                h_node = torch_checkpoint(
                    lambda _x, _i=kv_gather_idx, _m=kv_key_padding_mask: (
                        net_global[t_inter][0](
                            _x,
                            kv_gather_idx=_i,
                            kv_key_padding_mask=_m,
                        )
                    ),
                    h_node,
                    use_reentrant=False,
                )
            else:
                h_node = net_global[t_inter][0](
                    h_node,
                    kv_gather_idx=kv_gather_idx,
                    kv_key_padding_mask=kv_key_padding_mask,
                    external_kv=(
                        external_global_kv["node"]
                        if external_global_kv is not None
                        else None
                    ),
                )

            # --- feature channels ---
            if has_features:
                net_feat_idx = 0 if self.parameter_tying_across_channels else 1
                h_feature = self._apply_global_channel_attn(
                    net_global[t_inter][net_feat_idx],
                    h_feature,
                    bsize,
                    num_nodes,
                    kv_gather_idx,
                    kv_key_padding_mask,
                    (
                        external_global_kv["feature"]
                        if external_global_kv is not None
                        else None
                    ),
                )

            # --- label channels ---
            if has_labels:
                net_lab_idx = 0 if self.parameter_tying_across_channels else 2
                h_label = self._apply_global_channel_attn(
                    net_global[t_inter][net_lab_idx],
                    h_label,
                    bsize,
                    num_nodes,
                    kv_gather_idx,
                    kv_key_padding_mask,
                    (
                        external_global_kv["label"]
                        if external_global_kv is not None
                        else None
                    ),
                )
        self._profile_end(_prof_global)

        # --- Phase 3: structural random walk based update ---
        _prof_rw = self._profile_begin("rw", cuda=profile_cuda)
        _walks = walks_t.flatten(1, 2)
        _types = types_t.flatten(1, 2)

        rw_node_mask_now = torch.zeros(bsize, num_nodes, device=device, dtype=torch.bool)
        rw_node_mask_now.scatter_(1, _walks, True)

        rw_type_mask_now = torch.zeros(bsize, num_types + 1, device=device, dtype=torch.bool)
        rw_type_mask_now.scatter_(1, _types, True)
        
        
        if self.rw_update:
            # Outer Pre-LN: read from normalized states, write residuals to raw h_*.
            if self.use_pre_norm:
                i_pn = 0 if self.parameter_tying_across_refinements else t
                h_node_rd = self.pre_norm_node[i_pn](h_node)
                h_type_rd = self.pre_norm_type[i_pn](h_type)
                h_feature_rd = (
                    self.pre_norm_feature[i_pn](h_feature) if has_features else h_feature
                )
                h_label_rd = (
                    self.pre_norm_label[i_pn](h_label) if has_labels else h_label
                )
            else:
                h_node_rd = h_node
                h_type_rd = h_type
                h_feature_rd = h_feature
                h_label_rd = h_label

            walk_len = walks_t.shape[-1]
            # embedding [bsize, samples, len, dim]
            if self.embed_first_only and t > 0:
                x = torch.zeros(bsize, samples, walk_len, self.D, device=device) # shared walk embedding is skipped
            else:
                if head_index is not None: # for link prediction
                    is_h = torch.zeros(bsize, num_nodes, dtype=torch.long, device=device)
                    is_r = torch.zeros(bsize, num_types + 1, dtype=torch.long, device=device)
                    is_h[torch.arange(bsize, device=device), head_index] = 1
                    is_r[torch.arange(bsize, device=device), type_index] = 1
                    is_h = is_h.gather(1, _walks).view(bsize, samples, walk_len)
                    is_r = is_r.gather(1, _types).view(bsize, samples, walk_len)
                else: # for node classification
                    is_h = torch.full((bsize, samples, walk_len), 2, dtype=torch.long, device=device)
                    is_r = torch.full((bsize, samples, walk_len), 2, dtype=torch.long, device=device)

                i_emb = 0 if self.embedding_tying else t

                def rw_walk_embed_fn(
                    named_walks_t_: Tensor,
                    named_types_t_: Tensor,
                    directions_t_: Tensor,
                    is_h_: Tensor,
                    is_r_: Tensor,
                    h_type_n_: Tensor,
                    _types_: Tensor,
                ) -> Tensor:
                    x_ = (
                        self.emb_anon_node[i_emb](named_walks_t_ - 1)
                        + self.emb_anon_type[i_emb](named_types_t_ - 1)
                        + self.emb_node_is_query[i_emb](is_h_)
                        + self.emb_type_is_query[i_emb](is_r_)
                    )
                    if not (
                        self.add_inverse_edges_kgs
                        and self.only_forward_edges_when_inverses_added
                    ):
                        x_ = x_ + self.emb_direction[i_emb](directions_t_)
                    x_type_ = h_type_n_.gather(1, _types_[:, :, None].expand(-1, -1, self.D))
                    x_type_ = x_type_.view(bsize, samples, walk_len, self.D)
                    return x_ + self.from_type[t_inter](x_type_)

                if self.checkpoint_sublayers and torch.is_grad_enabled():
                    x = torch_checkpoint(
                        rw_walk_embed_fn,
                        named_walks_t,
                        named_types_t,
                        directions_t,
                        is_h,
                        is_r,
                        h_type_rd,
                        _types,
                        use_reentrant=False,
                    )
                else:
                    x = rw_walk_embed_fn(
                        named_walks_t,
                        named_types_t,
                        directions_t,
                        is_h,
                        is_r,
                        h_type_rd,
                        _types,
                    )

            if self.embed_first_only and t > 0:
                x_type = h_type_rd.gather(1, _types[:, :, None].expand(-1, -1, self.D))
                x_type = x_type.view(bsize, samples, walk_len, self.D)
                x = x + self.from_type[t_inter](x_type)

            # --- node channel ---
            x_node = h_node_rd.gather(1, _walks[:, :, None].expand(-1, -1, self.D))
            x_node = x_node.view(bsize, samples, walk_len, self.D)
            x_node_channel = self.from_node[t_inter](x_node)
            x_node_total = x_node_channel + self.walk_emb_gate_node * x
            x_node = x_node_total.view(bsize * samples, walk_len, self.D)
            if self.checkpoint_sublayers and torch.is_grad_enabled():
                x_node = torch_checkpoint(self.net_rw[t_inter][0], x_node, use_reentrant=False)
            else:
                x_node = self.net_rw[t_inter][0](x_node)

            # --- feature channels ---
            if has_features:
                net_feat_idx = 0 if self.parameter_tying_across_channels else 1
                s_feat, _ = self._process_attr_channel(
                    h_attr=h_feature_rd, x=x, _walks=_walks, walks_t=walks_t,
                    net=self.net_rw[t_inter][net_feat_idx],
                    from_proj=self.from_feature[t_inter],
                    to_proj=self.to_feature[t_inter],
                    logit_proj=self.feature_logit[t_inter] if self.attention_scatter else None,
                    num_nodes=num_nodes,
                    walk_emb_gate=self.walk_emb_gate_feature,
                    return_seq=False,
                )

            # --- label channel ---
            rw_labels = bool(has_labels)
            if rw_labels:
                net_lab_idx = 0 if self.parameter_tying_across_channels else 2
                s_lab, _ = self._process_attr_channel(
                    h_attr=h_label_rd, x=x, _walks=_walks, walks_t=walks_t,
                    net=self.net_rw[t_inter][net_lab_idx],
                    from_proj=self.from_label[t_inter],
                    to_proj=self.to_label[t_inter],
                    logit_proj=self.label_logit[t_inter] if self.attention_scatter else None,
                    num_nodes=num_nodes,
                    walk_emb_gate=self.walk_emb_gate_label,
                    return_seq=False,
                )

            # --- scatter: node + label (to graph nodes) ---
            if self.attention_scatter:
                s_node = consensus_softmax_stable(
                    self.to_node[t_inter](x_node), self.node_logit[t_inter](x_node),
                    walks_t, num_nodes,
                )
            else:
                s_node = consensus_mean(self.to_node[t_inter](x_node), walks_t, num_nodes)

            # --- relation (type) scatter (node channel only) ---
            if self.attention_scatter:
                s_type = consensus_softmax_stable(
                    self.to_type[t_inter](x_node),
                    self.type_logit[t_inter](x_node),
                    types_t,
                    num_types + 1,
                )
            else:
                s_type = consensus_mean(
                    self.to_type[t_inter](x_node),
                    types_t,
                    num_types + 1,
                )

            # --- state updates ---
            if self.additive_refinement:
                if self.layerscale_init > 0:
                    h_node = h_node + self.ls_node * s_node
                    h_type = h_type + self.ls_type * s_type
                    if has_features:
                        h_feature = h_feature + self.ls_feature * s_feat
                    if rw_labels:
                        h_label = h_label + self.ls_label * s_lab
                else:
                    h_node = h_node + s_node
                    h_type = h_type + s_type
                    if has_features:
                        h_feature = h_feature + s_feat
                    if rw_labels:
                        h_label = h_label + s_lab
            else:
                h_node = torch.where(rw_node_mask_now[:, :, None], s_node, h_node)
                h_type = torch.where(rw_type_mask_now[:, :, None], s_type, h_type)
                if has_features:
                    h_feature = torch.where(rw_node_mask_now[:, :, None, None], s_feat, h_feature)
                if rw_labels:
                    h_label = torch.where(rw_node_mask_now[:, :, None, None], s_lab, h_label)

        self._profile_end(_prof_rw)

        rw_node_mask = torch.logical_or(rw_node_mask, rw_node_mask_now)
        rw_type_mask = torch.logical_or(rw_type_mask, rw_type_mask_now)

        return h_node, h_feature, h_label, h_type, rw_node_mask, rw_type_mask

    def _link_prediction_node_roles(
        self,
        data_work: Data,
        query_index: Tensor,
        type_index: Tensor,
        predict_head: Tensor,
        num_graph_nodes: int,
    ) -> Tensor:
        """Per-row node role ids for link prediction.

        0 = other, 1 = query-relation-neighbor, 2 = query node. Neighbors are
        direction-aware: for tail prediction (``predict_head==0``) a node ``n`` is
        a neighbor if edge ``(query, r, n)`` exists; for head prediction
        (``predict_head==1``) if edge ``(n, r, query)`` exists. Computed on the
        already batch-edge-masked ``data_work`` graph, so held-out target edges
        are excluded (see ``query_loader._mask_batch_edges``).

        Returns:
            Long tensor ``[bsize, num_graph_nodes]``.
        """
        device = query_index.device
        bsize = query_index.shape[0]
        roles = torch.zeros(bsize, num_graph_nodes, dtype=torch.long, device=device)
        edge_index = data_work.edge_index
        edge_type = data_work.edge_type
        assert isinstance(edge_index, Tensor) and isinstance(edge_type, Tensor)
        src, dst = edge_index[0], edge_index[1]
        # single host sync for the small per-query tensors, then no per-row syncs
        q_list = query_index.tolist()
        r_list = type_index.tolist()
        ph_list = predict_head.tolist()
        for b in range(bsize):
            q, r, ph = q_list[b], r_list[b], ph_list[b]
            if ph == 0:
                m = (src == q) & (edge_type == r)
                nbrs = dst[m]
            else:
                m = (dst == q) & (edge_type == r)
                nbrs = src[m]
            roles[b, nbrs] = 1
            roles[b, q] = 2
        return roles

    @staticmethod
    def _data_for_cpu_walks(data: Data) -> Data:
        """Copy tensor fields to CPU for background walk generation."""
        out = Data()
        for key, value in data:
            if isinstance(value, Tensor):
                out[key] = value.detach().cpu()
            else:
                out[key] = value
        return out

    def _prefetch_data_cpu(self, data: Data) -> Data:
        """Reuse one CPU graph copy per forward for prefetch walk generation."""
        cache = getattr(self, "_walk_prefetch_cpu_cache", None)
        key = id(data)
        if cache is None or cache[0] != key:
            self._walk_prefetch_cpu_cache = (key, self._data_for_cpu_walks(data))
        return self._walk_prefetch_cpu_cache[1]

    def _generate_nc_walks(
        self, data: Data, batch_indices: Tensor, start_mode: str = "default",
    ) -> tuple:
        with self._profile_span("walks"):
            return self.walks_node_classification(
                self._prefetch_data_cpu(data),
                batch_indices.detach().cpu(),
                stay_on_cpu=True,
                start_mode=start_mode,
            )

    def _generate_lp_walks(self, data: Data, query_index: Tensor) -> tuple:
        return self._sample_link_prediction_walks(
            self._prefetch_data_cpu(data),
            query_index.detach().cpu(),
            stay_on_cpu=True,
        )

    def _sample_link_prediction_walks(
        self, data: Data, query_index: Tensor, stay_on_cpu: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Sample link-prediction walks on the full graph."""
        with self._profile_span("walks"):
            return self.walks_link_prediction(data, query_index, stay_on_cpu=stay_on_cpu)

    def _link_pred_query_embeddings(
        self,
        h_node: Tensor,
        h_type: Tensor,
        num_graph_nodes: int,
        type_index: Tensor,
        predict_head: Tensor,
        tail_index: Optional[Tensor],
    ) -> Tensor:
        """Pre-head query embeddings for candidate entities or all graph nodes."""
        bsize = type_index.shape[0]
        h_rel = h_type.gather(1, type_index[:, None, None].expand(bsize, 1, self.D))
        h_node_graph = h_node[:, :num_graph_nodes]

        if tail_index is not None:
            num_candidates = tail_index.shape[1]
            h_query = h_node_graph.gather(
                1, tail_index[:, :, None].expand(bsize, num_candidates, self.D)
            ) + h_rel
        else:
            h_query = h_node_graph + h_rel

        if not self.add_inverse_edges_kgs:
            h_dir = self.emb_direction[0](predict_head).unsqueeze(1)
            h_query = h_query + h_dir

        if self.additive_refinement:
            h_query = self.final_norm(h_query)
        return h_query

    def _link_pred_logits(
        self,
        h_node: Tensor,
        h_type: Tensor,
        num_graph_nodes: int,
        type_index: Tensor,
        predict_head: Tensor,
        tail_index: Optional[Tensor],
    ) -> Tensor:
        """Logits for candidate entities or all graph nodes."""
        h_query = self._link_pred_query_embeddings(
            h_node,
            h_type,
            num_graph_nodes,
            type_index,
            predict_head,
            tail_index,
        )
        return self.head(h_query).squeeze(-1).to(torch.float32)

    def _forward_link_prediction(
        self, data: Data, query_index: Tensor, type_index: Tensor,
        tail_index: Optional[Tensor] = None,
        predict_head: Optional[Tensor] = None,
    ) -> Tensor:
        walk_tuple = self._sample_link_prediction_walks(data, query_index, stay_on_cpu=False)
        return self._forward_link_prediction_from_walks(
            data, query_index, type_index, walk_tuple,
            tail_index=tail_index, predict_head=predict_head,
        )

    def _forward_link_prediction_from_walks(
        self,
        data: Data,
        query_index: Tensor,
        type_index: Tensor,
        walk_tuple: tuple,
        tail_index: Optional[Tensor] = None,
        predict_head: Optional[Tensor] = None,
    ) -> Tensor:
        """Link-prediction forward from an on-device walk tuple."""
        num_types = int(data.num_relations)
        device = query_index.device
        bsize = query_index.shape[0]
        if predict_head is None:
            predict_head = torch.zeros(bsize, dtype=torch.long, device=device)

        walks, named_walks, restarts, neighbors, types, named_types, directions = walk_tuple
        graph_visit_stats = (
            None if self.training
            else self.original_graph_visit_stats(walks, int(data.num_nodes))
        )
        if tail_index is not None:
            parts = [walks.flatten()]
            if self.record_neighbors:
                parts.append(neighbors.flatten())
            if self.disable_rw_graph_prune:
                keep_old = self._all_graph_node_indices(data, device)
            else:
                keep_old = torch.unique(
                    torch.cat([torch.cat(parts), query_index.reshape(-1), tail_index.reshape(-1)])
                )
            data_work, mapper = self._prune_graph_to_indices(data, keep_old)
            query_index = mapper[query_index]
            tail_index = mapper[tail_index]
            walks, named_walks, restarts, neighbors, types, named_types, directions = (
                self._remap_walk_tuple(walk_tuple, mapper)
            )
        else:
            data_work = data

        num_graph_nodes = data_work.num_nodes
        samples = walks.shape[2]
        _dummy = torch.tensor(0.0, device=device, requires_grad=False)

        # --- features ---
        features = data_work.x if hasattr(data_work, "x") else None
        has_real_features = features is not None and features.ndim == 2 and features.shape[1] > 0
        if has_real_features:
            features, _ = self._apply_feature_column_randomization(features)
            num_real_features = int(features.shape[1])
            if self.feature_groups > 1:
                num_real_features = math.ceil(num_real_features / self.feature_groups)
            h_feature = self.feature_embedding(features)
            if self.feature_init_norm is not None:
                h_feature = self.feature_init_norm(h_feature)
            h_feature = h_feature[None].expand(
                bsize, num_graph_nodes, num_real_features, self.D
            )
        else:
            num_real_features = 0
            h_feature = _dummy
        num_features = num_real_features + self.thinking_features
        has_features = num_features > 0
        h_feature = self._concat_thinking_features(h_feature, bsize, num_graph_nodes)
        if not has_features:
            h_feature = _dummy

        # --- labels ---
        labels = data_work.y if hasattr(data_work, "y") else None
        has_labels = labels is not None and labels.ndim == 2 and labels.shape[1] > 0
        if has_labels:
            num_label_classes = labels.shape[1]
            y_perm, _ = self._permute_label_columns(labels.long())
            h_label = self.label_embedding(y_perm)
            if self.label_init_norm is not None:
                h_label = self.label_init_norm(h_label)
            h_label = h_label[None].expand(
                bsize, num_graph_nodes, num_label_classes, self.D
            )
        else:
            num_label_classes = 0
            h_label = _dummy

        # --- node / type init (graph nodes only) ---
        node_roles = self._link_prediction_node_roles(
            data_work, query_index, type_index, predict_head, num_graph_nodes
        )
        h_node = self.node_role_emb(node_roles)
        h_type = self.type_init[None, None].expand(bsize, num_types + 1, self.D)

        h_node, h_feature, h_label = self._append_thinking_row_nodes(
            h_node,
            h_feature,
            h_label,
            bsize,
            num_features,
            has_features,
            has_labels,
            num_label_classes,
        )
        num_nodes = num_graph_nodes + self.thinking_rows

        rw_node_mask = torch.zeros(bsize, num_nodes, device=device, dtype=torch.bool)
        rw_type_mask = torch.zeros(bsize, num_types + 1, device=device, dtype=torch.bool)

        coverage_after_first = None

        # Restricted K/V mask for the global-attention branch:
        # KV nodes = query node + query-relation-neighbors (+ thinking rows).
        # Neighbor cap (if set) applies only here; node_roles are unchanged.
        if self.global_attention_update:
            global_kv_mask = self._lp_global_kv_mask_from_roles(
                node_roles, num_nodes,
            )
        else:
            global_kv_mask = None

        for t in range(self.T):
            step_args = (
                t, h_node, h_feature, h_label, h_type,
                rw_node_mask, rw_type_mask,
                walks[t], named_walks[t], restarts[t], neighbors[t],
                types[t], named_types[t], directions[t],
                samples, num_nodes, num_types,
                has_features, has_labels, bsize,
                query_index, type_index,
                global_kv_mask,
                None,  # external_global_kv
                None,  # capture_train_rows
                None,  # capture_out
                "link",
            )
            if (
                self.checkpoint_refinements
                and torch.is_grad_enabled()
                and not self.checkpoint_sublayers
            ):
                h_node, h_feature, h_label, h_type, rw_node_mask, rw_type_mask = \
                    torch_checkpoint(self._refinement_step, *step_args, use_reentrant=False)
            else:
                h_node, h_feature, h_label, h_type, rw_node_mask, rw_type_mask = \
                    self._refinement_step(*step_args)

            if not self.training and t == 0:
                coverage_after_first = {
                    "node": (rw_node_mask.float().mean(dim=1).mean() * 100).item(),
                    "type": (rw_type_mask[:, :num_types].float().mean(dim=1).mean() * 100).item(),
                }

        self._set_eval_coverage_stats(
            coverage_after_first, rw_node_mask, rw_type_mask, num_types,
            graph_visit_stats,
        )

        self._profile_mark_forward()
        return self._link_pred_logits(
            h_node,
            h_type,
            num_graph_nodes,
            type_index,
            predict_head,
            tail_index,
        )

    def _sample_real_nc_train_query_ids(
        self, data: Data, batch_indices: Tensor,
    ) -> Tensor:
        """Full-graph train query ids for real-world NC (includes the batch).

        Returns ``batch_indices`` when the extra-query expansion is off, at eval,
        or on synthetic / train-on-held-out graphs.
        """
        if not self.training or self._supervise_non_train_nodes(data):
            return batch_indices
        if self.nc_train_query_frac <= 0.0 or self.nc_train_query_cap <= 0:
            return batch_indices
        train_mask = getattr(data, "train_mask", None)
        if train_mask is None:
            return batch_indices
        train_idx = train_mask.nonzero(as_tuple=True)[0]
        if train_idx.numel() == 0:
            return batch_indices
        train_idx = train_idx.to(device=batch_indices.device, dtype=batch_indices.dtype)
        k = nc_train_query_count(
            int(train_idx.numel()),
            int(batch_indices.numel()),
            self.nc_train_query_frac,
            self.nc_train_query_cap,
        )
        return sample_nc_train_query_ids(train_idx, batch_indices, k)

    def _forward_node_classification(
        self,
        data: Data,
        batch_indices: Tensor,
        start_mode: str = "default",
    ) -> Tensor | tuple[Tensor, Tensor]:
        with self._profile_span("walks"):
            walk_tuple = self.walks_node_classification(
                data, batch_indices, stay_on_cpu=False, start_mode=start_mode,
            )
        return self._forward_node_classification_from_walks(
            data,
            batch_indices,
            walk_tuple,
        )

    def _forward_node_classification_from_walks(
        self,
        data: Data,
        batch_indices: Tensor,
        walk_tuple: tuple,
        feat_perm: Optional[Tensor] = None,
        label_perm: Optional[Tensor] = None,
        cached_kv: Optional[list] = None,
        capture_train_ids: Optional[Tensor] = None,
        capture_out: Optional[list] = None,
        train_query_ids: Optional[Tensor] = None,
    ) -> Tensor | tuple:
        """NC forward from an on-device walk tuple (prune through head).

        Cached train-KV eval extensions (all default to None = current path):
            feat_perm / label_perm: replay explicit column permutations.
            cached_kv: materialized per-refinement external global-attention
                K/V (Phase B); replaces the live train-node K/V gather.
            capture_train_ids / capture_out: global train node ids to force
                into the pruned graph and snapshot before global attention at
                every refinement step (Phase A); snapshots are appended to
                ``capture_out`` as per-step channel dicts.

            train_query_ids: optional full-graph node ids to hide / supervise on
                real-world NC training (must include the batch). When unset, a
                set is sampled from ``--nc_train_query_frac`` / cap.
        """
        num_types = int(data.num_relations)
        device = batch_indices.device
        bsize = 1

        if cached_kv is not None or capture_out is not None:
            assert self.thinking_rows == 0, (
                "cached train-KV eval does not support thinking rows"
            )

        assert data.y is not None and data.y.ndim == 2, "When we do node classification, we need to have labels"
        num_classes = data.y.shape[1]
        effective_num_classes = num_classes

        if train_query_ids is None:
            train_query_ids = self._sample_real_nc_train_query_ids(data, batch_indices)
        else:
            train_query_ids = train_query_ids.to(device=device, dtype=torch.long)

        walks, named_walks, restarts, neighbors, types, named_types, directions = walk_tuple
        graph_visit_stats = (
            None if self.training
            else self.original_graph_visit_stats(walks, int(data.num_nodes))
        )
        parts = [walks.flatten()]
        if self.record_neighbors:
            parts.append(neighbors.flatten())
        if self.disable_rw_graph_prune:
            keep_old = self._all_graph_node_indices(data, device)
        else:
            keep_parts = [torch.cat(parts), batch_indices.reshape(-1)]
            if capture_train_ids is not None:
                # Phase A: every train node must be present in the pruned
                # graph so each pass yields a complete KV bank.
                keep_parts.append(capture_train_ids.to(device).reshape(-1))
            keep_old = torch.unique(torch.cat(keep_parts))
        data_work, mapper = self._prune_graph_to_indices(data, keep_old)
        batch_indices = mapper[batch_indices]
        query_local = mapper[train_query_ids]
        query_local = query_local[query_local >= 0]
        if query_local.numel() == 0:
            query_local = batch_indices
        capture_train_rows = (
            mapper[capture_train_ids.to(device)]
            if capture_train_ids is not None
            else None
        )
        walks, named_walks, restarts, neighbors, types, named_types, directions = self._remap_walk_tuple(
            walk_tuple, mapper
        )

        num_graph_nodes = data_work.num_nodes
        train_mask = data_work.train_mask
        samples = walks.shape[2]
        _dummy = torch.tensor(0.0, device=device, requires_grad=False) # if no features, use this dummy tensor

        # --- features ---
        features = data_work.x if hasattr(data_work, "x") else None
        has_real_features = features is not None and features.ndim == 2 and features.shape[1] > 0
        if has_real_features:
            features, feat_perm = self._apply_feature_column_randomization(
                features, perm=feat_perm,
            )
            num_real_features = int(features.shape[1])
            if self.feature_groups > 1:
                num_real_features = math.ceil(num_real_features / self.feature_groups)
            h_feature = self.feature_embedding(features)
            if self.feature_init_norm is not None:
                h_feature = self.feature_init_norm(h_feature)
            h_feature = h_feature[None].expand(
                bsize, num_graph_nodes, num_real_features, self.D
            )
        else:
            num_real_features = 0
            h_feature = _dummy
        num_features = num_real_features + self.thinking_features
        has_features = num_features > 0
        h_feature = self._concat_thinking_features(h_feature, bsize, num_graph_nodes)
        if not has_features:
            h_feature = _dummy

        # --- labels ---
        y_perm, label_perm = self._permute_label_columns(
            data_work.y.long(), perm=label_perm,
        )
        y_oh = y_perm[None].clone()
        y_oh[0, ~train_mask, :] = 2
        y_oh[0, query_local, :] = 2
        h_label = self.label_embedding(y_oh)
        if self.label_init_norm is not None:
            h_label = self.label_init_norm(h_label)
        has_labels = True

        # --- node / type init (graph nodes only) ---
        h_node = self.node_role_emb.weight[0][None, None].expand(
            bsize, num_graph_nodes, self.D
        )
        h_type = self.type_init[None, None].expand(bsize, num_types + 1, self.D)

        h_node, h_feature, h_label = self._append_thinking_row_nodes(
            h_node,
            h_feature,
            h_label,
            bsize,
            num_features,
            has_features,
            has_labels,
            effective_num_classes,
        )
        self._log_wander_mem(
            num_nodes=num_graph_nodes + self.thinking_rows,
            has_features=has_features,
            has_labels=has_labels,
            h_node=h_node,
            h_feature=h_feature,
            h_label=h_label,
            global_kv_mask=None,
            tag="pre_materialize",
        )
        h_node, h_feature, h_label = self._materialize_token_states(
            h_node, h_feature, h_label, has_features, has_labels,
        )
        num_nodes = num_graph_nodes + self.thinking_rows

        rw_node_mask = torch.zeros(bsize, num_nodes, device=device, dtype=torch.bool)
        rw_type_mask = torch.zeros(bsize, num_types + 1, device=device, dtype=torch.bool)

        coverage_after_first = None

        train_mask_ext = self._extend_train_mask_for_thinking_rows(
            train_mask, self.thinking_rows, device
        )

        # Restricted K/V mask for the global-attention branch:
        # KV nodes = train_mask AND not in this pass's batch (query) nodes.
        # With cached_kv (Phase B) the K/V come pre-projected from the train-KV
        # cache with batch nodes already excluded (see _ensemble_nc_eval_cached).
        if self.global_attention_update and cached_kv is None:
            in_batch = torch.zeros(bsize, num_nodes, dtype=torch.bool, device=device)
            in_batch[0, query_local] = True
            global_kv_mask = train_mask_ext[None].expand(bsize, -1) & ~in_batch
        else:
            global_kv_mask = None

        self._log_wander_mem(
            num_nodes=num_nodes,
            has_features=has_features,
            has_labels=has_labels,
            h_node=h_node,
            h_feature=h_feature,
            h_label=h_label,
            global_kv_mask=global_kv_mask,
            tag="post_materialize",
        )

        for t in range(self.T):
            step_args = (
                t, h_node, h_feature, h_label, h_type,
                rw_node_mask, rw_type_mask,
                walks[t], named_walks[t], restarts[t], neighbors[t],
                types[t], named_types[t], directions[t],
                samples, num_nodes, num_types,
                has_features, has_labels, bsize,
                None, None,
                global_kv_mask,
                cached_kv[t] if cached_kv is not None else None,
                capture_train_rows,
                capture_out,
                "node",
            )
            if (
                self.checkpoint_refinements
                and torch.is_grad_enabled()
                and not self.checkpoint_sublayers
            ):
                h_node, h_feature, h_label, h_type, rw_node_mask, rw_type_mask = \
                    torch_checkpoint(self._refinement_step, *step_args, use_reentrant=False)
            else:
                h_node, h_feature, h_label, h_type, rw_node_mask, rw_type_mask = \
                    self._refinement_step(*step_args)

            if not self.training and t == 0:
                coverage_after_first = {
                    "node": (rw_node_mask.float().mean(dim=1).mean() * 100).item(),
                    "type": (rw_type_mask[:, :num_types].float().mean(dim=1).mean() * 100).item(),
                }

        self._set_eval_coverage_stats(
            coverage_after_first, rw_node_mask, rw_type_mask, num_types,
            graph_visit_stats,
        )

        # Phase A capture-only pass (cached train-KV eval): the snapshots in
        # ``capture_out`` are the product; no head / logits needed.
        if capture_out is not None:
            return None

        self._profile_mark_forward()

        # Dataset-type heuristic must use the full graph, not the pruned
        # subgraph (see ``_supervise_non_train_nodes``).
        supervise_non_train = self._supervise_non_train_nodes(data)

        # Synthetic / train-on-held-out: supervise all surviving val∪test nodes.
        if self.training and supervise_non_train:
            test_idx_local = self._non_train_supervise_idx(
                data_work, num_graph_nodes=num_graph_nodes
            )
            h_out = h_label[0, test_idx_local, :, :]
            if self.additive_refinement:
                h_out = self.final_norm(h_out)
            logits = self.node_cls_head_one_hot(h_out).squeeze(-1)
            inv_perm = torch.argsort(label_perm)
            unpermuted = logits[:, inv_perm]

            targets = data_work.y[test_idx_local].argmax(dim=1)
            return unpermuted.to(torch.float32), targets

        h_batch = h_label[0, query_local if self.training else batch_indices, :, :]
        if self.additive_refinement:
            h_batch = self.final_norm(h_batch)
        logits = self.node_cls_head_one_hot(h_batch).squeeze(-1)
        logits = logits.reshape(-1, effective_num_classes)
        inv_perm = torch.argsort(label_perm)
        logits = logits[:, inv_perm]

        if self.training:
            targets = data_work.y[query_local].argmax(dim=1)
            return logits.to(torch.float32), targets

        return logits.to(torch.float32)

    # ------------------------------------------------------------------
    # Cached train-KV eval (two-phase NC eval)
    # ------------------------------------------------------------------

    def attach_train_kv_cache(self, cache: Optional[TrainKVCache]) -> None:
        """Attach (or detach with ``None``) the per-dataset train-KV cache."""
        if self._train_kv_cache is not None and cache is not self._train_kv_cache:
            self._train_kv_cache.release_materialized()
        self._train_kv_cache = cache

    @torch.no_grad()
    def precompute_train_kv_cache(
        self, data: Data, num_passes: Optional[int] = None,
    ) -> TrainKVCache:
        """Phase A: cache pre-global-attention train-node hidden states.

        Each pass samples its own feature/label column permutations (stored for
        replay in Phase B). Walk starts are stratified over train-node chunks of
        size ``train_kv_chunk_size`` (``train_chunk`` mode); ``batch_indices``
        stays empty so train nodes remain K/V contributors, not query nodes.
        Snapshots from each chunk forward are merged by owner chunk into one
        pass bank of shape ``[M, …]`` per refinement step.
        """
        assert not self.training, "precompute_train_kv_cache is eval-only"
        assert self.global_attention_update, (
            "cached train-KV eval requires --wander_global_attention_update"
        )
        assert self.thinking_rows == 0, (
            "cached train-KV eval does not support thinking rows"
        )
        if num_passes is None:
            num_passes = self.train_kv_passes

        device = data.edge_index.device
        train_mask = getattr(data, "train_mask", None)
        assert train_mask is not None and int(train_mask.sum().item()) > 0, (
            "cached train-KV eval requires a non-empty train_mask"
        )
        num_features = self._num_features_for_walk_cap(data)
        eff_N, eff_L = self._effective_walk_params(
            int(data.num_nodes), num_features, base_n=self.train_kv_walk_num,
        )
        walk_note = (
            f"walk_num={eff_N} (base={self.train_kv_walk_num}, "
            f"eval_base={self._walk_num_for_start_mode('batch_only')}), "
            f"walk_len={eff_L}"
        )

        train_ids = train_mask.nonzero(as_tuple=True)[0]
        m = int(train_ids.numel())
        chunk_size = int(self.train_kv_chunk_size)
        if chunk_size <= 0 or chunk_size >= m:
            train_chunks = [train_ids]
        else:
            train_chunks = [
                train_ids[i : i + chunk_size] for i in range(0, m, chunk_size)
            ]
        n_chunks = len(train_chunks)
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f"[train_kv_cache] Phase A: {num_passes} passes x {n_chunks} train chunks "
                f"(chunk_size={chunk_size}, M={m}, {walk_note})",
                flush=True,
            )

        cache = TrainKVCache(train_ids=train_ids.detach().cpu())
        empty_batch = torch.empty(0, dtype=torch.long, device=device)
        id_to_col = {int(nid.item()): i for i, nid in enumerate(train_ids)}

        for _ in range(num_passes):
            feat_perm = None
            if self.randomize_feat_columns and self._data_has_real_features(data):
                feat_perm = torch.randperm(int(data.x.shape[1]), device=device)
            label_perm = None
            if self.randomize_label_columns and int(data.y.shape[1]) >= 2:
                label_perm = torch.randperm(int(data.y.shape[1]), device=device)

            merged_hidden: Optional[list[dict[str, Tensor]]] = None
            for chunk in train_chunks:
                chunk = chunk.to(device=device)
                walk_tuple = self.walks_node_classification(
                    data,
                    empty_batch,
                    stay_on_cpu=False,
                    start_mode="train_chunk",
                    walk_focus_indices=chunk,
                )
                capture: list = []
                self._forward_node_classification_from_walks(
                    data, empty_batch, walk_tuple,
                    feat_perm=feat_perm, label_perm=label_perm,
                    capture_train_ids=train_ids, capture_out=capture,
                )
                assert len(capture) == self.T, (
                    f"expected {self.T} capture steps, got {len(capture)}"
                )
                chunk_cols = torch.tensor(
                    [id_to_col[int(n.item())] for n in chunk],
                    dtype=torch.long,
                    device=device,
                )
                if merged_hidden is None:
                    merged_hidden = [
                        {
                            name: torch.empty_like(h)
                            for name, h in step.items()
                        }
                        for step in capture
                    ]
                for t, step_cap in enumerate(capture):
                    for name, h in step_cap.items():
                        merged_hidden[t][name].index_copy_(
                            0, chunk_cols, h.index_select(0, chunk_cols),
                        )

            assert merged_hidden is not None
            hidden = [
                {
                    name: h.to(device="cpu", dtype=self.train_kv_cache_dtype)
                    for name, h in step.items()
                }
                for step in merged_hidden
            ]
            cache.add_pass(feat_perm, label_perm, hidden)

        cache.log_size()
        return cache

    def _ensemble_nc_eval_cached(
        self, data: Data, batch_indices: Tensor,
    ) -> Tensor:
        """Phase B: batch-only walks + cached train K/V in global attention.

        Draw ``i`` uses cached pass ``i % num_passes`` (its K/V and its
        feature/label permutations). Query nodes in ``batch_indices`` are
        dropped from the materialized K/V so train_eval matches the live
        path (``train_mask & ~in_batch``). Supports both the sequential and
        the WalkPrefetcher-pipelined ensemble paths.
        """
        cache = self._train_kv_cache
        assert cache is not None and cache.num_passes > 0
        device = batch_indices.device

        def forward_with_pass(walk_tuple: tuple, draw_idx: int) -> Tensor:
            pass_idx = draw_idx % cache.num_passes
            cached_pass = cache.passes[pass_idx]
            kv = TrainKVCache.exclude_nodes(
                cache.materialize(pass_idx, self, device),
                cache.train_ids,
                batch_indices,
            )
            return self._forward_node_classification_from_walks(
                data, batch_indices, walk_tuple,
                feat_perm=cached_pass.feat_perm,
                label_perm=cached_pass.label_perm,
                cached_kv=kv,
            )

        mode = getattr(self, "_ensemble_eval_mode", None)
        if mode == "sequential" or not self.use_ensemble_prefetch:
            logits = []
            for i in range(self.test_samples):
                walk_tuple = self.walks_node_classification(
                    data, batch_indices, stay_on_cpu=False,
                    start_mode="batch_only",
                )
                logits.append(forward_with_pass(walk_tuple, i))
            return torch.stack(logits, dim=0).mean(0)

        draw_counter = [0]

        def forward_from_walks(walk_tuple: tuple) -> Tensor:
            i = draw_counter[0]
            draw_counter[0] += 1
            return forward_with_pass(walk_tuple, i)

        outputs = self._pipelined_ensemble_from_walks(
            lambda: self._generate_nc_walks(
                data, batch_indices, start_mode="batch_only",
            ),
            device,
            forward_from_walks,
        )
        return torch.stack(outputs, dim=0).mean(0)

    def _ensemble_nc_eval_sequential(
        self,
        data: Data,
        batch_indices: Tensor,
    ) -> Tensor:
        logits = []
        for _ in range(self.test_samples):
            walk_tuple = self.walks_node_classification(
                data, batch_indices, stay_on_cpu=False,
            )
            logits.append(
                self._forward_node_classification_from_walks(
                    data, batch_indices, walk_tuple,
                )
            )
        return torch.stack(logits, dim=0).mean(0)

    def _ensemble_nc_eval_pipelined(
        self,
        data: Data,
        batch_indices: Tensor,
    ) -> Tensor:
        def forward_from_walks(walk_tuple: tuple):
            return self._forward_node_classification_from_walks(
                data, batch_indices, walk_tuple,
            )

        outputs = self._pipelined_ensemble_from_walks(
            lambda: self._generate_nc_walks(data, batch_indices),
            batch_indices.device,
            forward_from_walks,
        )
        return torch.stack(outputs, dim=0).mean(0)

    def _ensemble_nc_eval(
        self,
        data: Data,
        batch_indices: Tensor,
    ) -> Tensor:
        if self._train_kv_cache is not None:
            return self._ensemble_nc_eval_cached(data, batch_indices)
        mode = getattr(self, "_ensemble_eval_mode", None)
        if mode == "sequential" or not self.use_ensemble_prefetch:
            return self._ensemble_nc_eval_sequential(data, batch_indices)
        return self._ensemble_nc_eval_pipelined(data, batch_indices)

    def _lp_use_batched_ensemble(self, data: Data) -> bool:
        mode = getattr(self, "_ensemble_eval_mode", None)
        if mode == "lp_batched":
            return True
        if mode == "lp_batched_legacy_gate" or getattr(
            self, "_force_lp_legacy_ensemble_gate", False,
        ):
            return False
        return not self._needs_per_draw_randomization(data)

    def _forward_link_prediction_batched_ensemble(
        self,
        data: Data,
        query_index: Tensor,
        type_index: Tensor,
        tail_index: Optional[Tensor] = None,
        predict_head: Optional[Tensor] = None,
    ) -> Tensor:
        bsize = query_index.shape[0]
        query_index = query_index.repeat_interleave(self.test_samples, dim=0)
        type_index = type_index.repeat_interleave(self.test_samples, dim=0)
        predict_head = predict_head.repeat_interleave(self.test_samples, dim=0)
        if tail_index is not None:
            tail_index = tail_index.repeat_interleave(self.test_samples, dim=0)
        logits = self._forward_link_prediction(
            data, query_index, type_index,
            tail_index=tail_index, predict_head=predict_head,
        )
        num_scores = logits.shape[1]
        return logits.view(bsize, self.test_samples, num_scores).mean(1)

    def _ensemble_lp_eval_sequential(
        self,
        data: Data,
        query_index: Tensor,
        type_index: Tensor,
        tail_index: Optional[Tensor] = None,
        predict_head: Optional[Tensor] = None,
    ) -> Tensor:
        logits = []
        for _ in range(self.test_samples):
            walk_tuple = self._sample_link_prediction_walks(
                data, query_index, stay_on_cpu=False,
            )
            logits.append(
                self._forward_link_prediction_from_walks(
                    data, query_index, type_index, walk_tuple,
                    tail_index=tail_index, predict_head=predict_head,
                )
            )
        return torch.stack(logits, dim=0).mean(0)

    def _ensemble_lp_eval_pipelined(
        self,
        data: Data,
        query_index: Tensor,
        type_index: Tensor,
        tail_index: Optional[Tensor] = None,
        predict_head: Optional[Tensor] = None,
    ) -> Tensor:
        def forward_from_walks(walk_tuple: tuple):
            return self._forward_link_prediction_from_walks(
                data, query_index, type_index, walk_tuple,
                tail_index=tail_index, predict_head=predict_head,
            )

        outputs = self._pipelined_ensemble_from_walks(
            lambda: self._generate_lp_walks(data, query_index),
            query_index.device,
            forward_from_walks,
        )
        return torch.stack(outputs, dim=0).mean(0)

    def _ensemble_lp_eval(
        self,
        data: Data,
        query_index: Tensor,
        type_index: Tensor,
        tail_index: Optional[Tensor] = None,
        predict_head: Optional[Tensor] = None,
    ) -> Tensor:
        if self._lp_use_batched_ensemble(data):
            return self._forward_link_prediction_batched_ensemble(
                data, query_index, type_index,
                tail_index=tail_index, predict_head=predict_head,
            )
        mode = getattr(self, "_ensemble_eval_mode", None)
        if mode == "sequential" or not self.use_ensemble_prefetch:
            return self._ensemble_lp_eval_sequential(
                data, query_index, type_index,
                tail_index=tail_index, predict_head=predict_head,
            )
        return self._ensemble_lp_eval_pipelined(
            data, query_index, type_index,
            tail_index=tail_index, predict_head=predict_head,
        )

    def forward(
        self, batch: Tensor, data: Data, candidate_tails: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Arguments:
            batch: [batch_size, 3] or [batch_size, 4] with columns
                (head, relation, tail[, predict_head]) for link prediction, OR
                [batch_size] node indices for node classification.
            data: PyG Data with edge_index, edge_type, num_relations, num_nodes.
            candidate_tails: optional [batch_size, num_candidates] (link prediction only).

        Returns:
            Link prediction with candidate_tails: [batch_size, num_candidates]
            Link prediction without candidate_tails: [batch_size, num_nodes]
            Node classification: [batch_size, num_classes]
        """
        if batch.ndim == 1: # node classification
            if self.training:
                return self._forward_node_classification(data, batch)
            if self._train_kv_cache is not None:
                # Cached train-KV eval handles any test_samples (incl. 1).
                return self._ensemble_nc_eval(data, batch)
            if self.test_samples == 1:
                return self._forward_node_classification(data, batch)
            return self._ensemble_nc_eval(data, batch)

        type_index = batch[:, 1]
        if batch.shape[1] >= 4:
            predict_head = batch[:, 3].long()
        else:
            predict_head = torch.zeros(batch.shape[0], dtype=torch.long, device=batch.device)
        query_index = torch.where(predict_head.bool(), batch[:, 2], batch[:, 0])

        if self.training or self.test_samples == 1:
            return self._forward_link_prediction(
                data, query_index, type_index,
                tail_index=candidate_tails, predict_head=predict_head,
            )

        return self._ensemble_lp_eval(
            data, query_index, type_index,
            tail_index=candidate_tails, predict_head=predict_head,
        )
