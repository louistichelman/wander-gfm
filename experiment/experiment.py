"""Wander training and evaluation driver.

main.py constructs Experiment and calls run_many_seeds. That loops over seeds,
load_datasets, then run_single_seed (wandb, model, init / resume). --eval_only
scores each test graph with evaluate. Otherwise trainer runs train_epoch
(_train_step_multi) and either logs train loss only (--skip_eval, async eval in
checkpoint_watcher) or calls run_training_epoch_eval -> evaluate /
evaluate_synthetic. After training, run_final_test_eval scores the best
checkpoint on test.

evaluate dispatches LP protocol (filtered MRR, Recall@K, sampled Recall) or
native node classification. The watcher reuses load_datasets,
run_training_epoch_eval, run_final_test_eval, and load_checkpoint_model_state
without going through run_many_seeds.
"""

import math
import os
import shutil
import sys
import importlib.util
import traceback
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import Tensor
from torch_geometric.data import Data
import wandb

from config import SEEDS
from config.arguments import (
    build_wander_config,
    effective_eval_batch_size_link,
    effective_eval_batch_size_node,
    prior_fixed_hp_overrides,
)
from data import DataSet, get_datasetargs, GraphBundle
from data.graph_bundle import (
    apply_filtered_entity_mask_,
    get_eval_filter_index,
    get_train_filter_index,
    is_link_prediction,
    resolve_split,
)
from data.lp_rank_stats import (
    RankAccum,
    is_per_relation_metric,
    relation_names_from_bundle,
    should_log_metric_to_wandb,
)

def _print_bundle_stats(name: str, bundle: GraphBundle) -> None:
    """Print summary stats for a loaded GraphBundle."""
    data = resolve_split(bundle, "train")
    print(f"{name}: \n  {data.num_nodes} nodes, \n  {data.edge_index.shape[1]} edges")
    if bundle.is_inductive:
        print(f"  inductive bundle (filter={bundle.inductive_filter_mode})")
    if hasattr(data, "x") and data.x is not None:
        print(f"  {data.x.shape[1]} node features")
    n_rel = getattr(data, "num_relations", None)
    if n_rel is not None:
        print(f"  {int(n_rel)} relation type(s)")
    if hasattr(data, "y") and data.y is not None:
        if data.y.ndim == 2:
            print(f"  {data.y.shape[1]} node label classes (one-hot)")
        else:
            print(f"  {data.y.max().item() + 1} node labels")
    for split in ("train", "val", "test"):
        try:
            split_data = resolve_split(bundle, split)  # type: ignore[arg-type]
            if is_link_prediction(split_data):
                mask = getattr(split_data, f"{split}_mask")
                if mask is not None:
                    print(f"  {mask.sum()} {split} edges")
            else:
                mask = getattr(split_data, f"{split}_mask", None)
                if mask is not None:
                    print(f"  {mask.sum()} {split} nodes")
        except (AttributeError, ValueError):
            pass


def _eval_progress_total(
    query_loader, *, is_native_node_cls: bool,
) -> int:
    """Number of eval queries (LP) or nodes (NC) to score."""
    if is_native_node_cls:
        global_count = getattr(query_loader, "eval_scoring_node_count_global", None)
        if callable(global_count):
            return int(global_count())
        return int(query_loader.eval_scoring_node_indices().numel())
    num_queries = getattr(query_loader, "num_queries", None)
    if num_queries is None:
        return 0
    perm = getattr(query_loader, "_eval_query_perm", None)
    if perm is not None:
        return int(perm.numel())
    max_eval = getattr(query_loader, "max_eval_samples", None)
    if max_eval is not None:
        return min(int(max_eval), int(num_queries))
    return int(num_queries)


def _log_eval_progress(
    *,
    dataset_name: str,
    split: str,
    is_native_node_cls: bool,
    batch_idx: int,
    items_done: int,
    total_items: int,
    correct_count: int,
    total_count: int,
    reciprocal_rank_sum: float,
    rank_count: int,
) -> None:
    """Periodic eval progress (NC: every 10 batches, LP: every 50)."""
    interval = 10 if is_native_node_cls else 50
    batches_done = batch_idx + 1
    if batches_done % interval != 0:
        return
    unit = "nodes" if is_native_node_cls else "queries"
    remaining = max(total_items - items_done, 0)
    msg = (
        f"[eval progress] {dataset_name} ({split}): "
        f"{items_done}/{total_items} {unit} ({remaining} remaining), "
        f"batch {batches_done}"
    )
    if is_native_node_cls and total_count > 0:
        msg += f", accuracy={correct_count / total_count:.4f}"
    elif rank_count > 0:
        msg += f", mrr={reciprocal_rank_sum / rank_count:.4f}"
    print(msg, flush=True)


from models import Wander
from models.wander.rw_checkpoint_keys import (
    filter_obsolete_checkpoint_keys,
    merge_init_checkpoint_state_dict,
    rw_parameter_keys,
)

from .eval_batch_resume import (
    apply_resume_payload,
    build_eval_batch_fingerprint,
    cat_chunks,
    clear_eval_batch_resume,
    eval_batch_resume_enabled_from_env,
    eval_batch_resume_path,
    eval_preempt_flag,
    resolve_and_validate_resume,
    save_eval_batch_resume,
)
from .utils import (
    set_seed,
    capture_rng_state as _capture_rng_state,
    restore_rng_state as _restore_rng_state,
    seed_rngs as _seed_rngs,
    generate_experiment_name,
    get_wandb_run_name,
    get_checkpoint_dir,
    get_checkpoint_path,
    read_prior_complexity_override,
    save_training_args,
    save_eval_run_args,
    save_wandb_run_id,
    load_wandb_run_id,
    save_json,
    save_epoch_eval_metrics,
    save_best_val_metrics,
    save_final_test_metrics,
    write_training_status,
    define_async_wandb_metrics,
)
from .query_loader import (
    MultiGraphLoader,
    QueryLoaderRecallAtK,
    QueryLoaderSampledRecall,
)
from .negative_sampling import sample_negative_entities
from .loss_and_metrics import (
    compute_link_prediction_loss,
    compute_node_classification_loss,
    macro_ovr_roc_auc_score,
    aggregate_metrics, aggregate_seed_results, print_aggregated_results,
)

from data.prior.dataset import SCMPrior
from data.prior.prior_config import get_default_fixed_hp, get_default_sampled_hp
from data.prior.prior_config_loader import load_prior_config, prior_config_module_path
from .synthetic_loader import SyntheticGraphLoader, close_synthetic_prefetch
from .mixed_loader import MixedTrainLoader

class Experiment:
    """Main experiment class for training and evaluating on multiple graphs."""
    
    def __init__(self, args, rank: int = 0, world_size: int = 1):
        """Initialize the experiment.
        
        Args:
            args: Parsed command-line arguments.
            rank: Local GPU rank (0 for single-GPU).
            world_size: Total number of GPUs (1 for single-GPU).
        """
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        
        self.amp_dtype = getattr(torch, args.amp_dtype)

        def _resolve_base_name() -> str:
            if getattr(args, "run_name", None):
                return args.run_name
            return generate_experiment_name(args)

        # DDP: each process has its own Python random state, so generate_experiment_name()
        # would pick different 2-char suffixes per rank. Only rank 0 creates the checkpoint
        # directory, so other ranks must use the same base_name (broadcast from rank 0).
        if dist.is_initialized() and world_size > 1:
            if self.is_main:
                name_list = [_resolve_base_name()]
            else:
                name_list = [None]
            dist.broadcast_object_list(
                name_list,
                src=0,
                device=self.device if torch.cuda.is_available() else None,
            )
            self.base_name = name_list[0]
        else:
            self.base_name = _resolve_base_name()
        self.checkpoint_dir = get_checkpoint_dir(args.save_load_path, self.base_name)
        
        if self.is_main:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        if dist.is_initialized() and world_size > 1:
            dist.barrier()
        if self.is_main:
            print(f"Experiment name: {self.base_name}")
            print(f"Checkpoint directory: {self.checkpoint_dir}")
            print(f"World size: {world_size}")

        if getattr(args, "synthetic_prior", False):
            self._prior_config_name = getattr(args, "prior_config", "default")
            self._bind_prior_config(self._prior_config_name)
            if self.is_main:
                print(f"prior_config={self._prior_config_name}")
                native = get_default_fixed_hp(0.0).get("native_backend")
                if native:
                    print(
                        f"native_backend={native} "
                        "(node_cls + undirected LP from official prior graphs)"
                    )
            self._restore_and_save_prior_config()

    def _bind_prior_config(self, name: str) -> None:
        """Bind module-level prior HP getters to the selected config variant."""
        global get_default_fixed_hp, get_default_sampled_hp
        mod = load_prior_config(name)
        get_default_fixed_hp = mod.get_default_fixed_hp
        get_default_sampled_hp = mod.get_default_sampled_hp

    def _use_direction_head_queries(self) -> bool:
        return not getattr(self.args, "add_inverse_edges_KGs", True)

    def _evaluate_head_predictions(self):
        """CLI override for head-prediction eval, or None to use dataset preference."""
        return getattr(self.args, "evaluate_head_predictions", None)

    def _link_pred_eval(self):
        """CLI override for the LP eval protocol, or None to use dataset preference."""
        return getattr(self.args, "link_pred_eval", None)

    def _eval_seeds(self) -> list[int]:
        seeds = getattr(self.args, "seeds", None)
        if seeds:
            return [int(s) for s in seeds]
        return list(SEEDS)

    def _recall_k(self) -> int:
        return int(getattr(self.args, "recall_k", 20))

    def _eval_batch_resume_enabled(self) -> bool:
        return eval_batch_resume_enabled_from_env(self.args)

    def _eval_resume_n_items(self, query_loader) -> int:
        global_count = getattr(query_loader, "eval_scoring_node_count_global", None)
        if callable(global_count):
            return int(global_count())
        scoring = getattr(query_loader, "eval_scoring_node_indices", None)
        if callable(scoring):
            return int(scoring().numel())
        return 0

    def _maybe_stop_eval_on_preempt(
        self,
        flag: Dict[str, bool],
        *,
        dataset_name: str,
        split: str,
        resume_path: str,
    ) -> None:
        stop = bool(flag.get("stop"))
        if self.world_size > 1:
            token = torch.tensor(
                [1 if stop else 0], dtype=torch.long, device=self.device
            )
            dist.all_reduce(token, op=dist.ReduceOp.MAX)
            stop = bool(token.item())
        if not stop:
            return
        if self.is_main:
            print(
                f"[eval resume] {dataset_name} ({split}): preempted; "
                f"saved progress to {resume_path}",
                flush=True,
            )
        raise SystemExit(143)
    
    def _restore_and_save_prior_config(self):
        """Restore prior_config.py from a resumed checkpoint (if available), then
        save it into the current run's checkpoint directory for reproducibility."""
        global get_default_fixed_hp, get_default_sampled_hp

        resumed_prior_config = None
        if self.args.resume_checkpoint:
            resume_dir = os.path.dirname(self.args.resume_checkpoint)
            candidate = os.path.join(resume_dir, "prior_config.py")
            if os.path.isfile(candidate):
                resumed_prior_config = candidate
                spec = importlib.util.spec_from_file_location(
                    "data.prior.prior_config", candidate,
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                sys.modules["data.prior.prior_config"] = mod
                get_default_fixed_hp = mod.get_default_fixed_hp
                get_default_sampled_hp = mod.get_default_sampled_hp
                if self.is_main:
                    print(f"Restored prior_config from {candidate}")

        if self.is_main:
            src = resumed_prior_config or prior_config_module_path(
                getattr(self, "_prior_config_name", "default")
            )
            dst = os.path.join(self.checkpoint_dir, "prior_config.py")
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.copy2(src, dst)
            print(f"Saved prior_config.py to {dst}")

    def _load_dataset_safe(self, dataset: DataSet, data_dir: str, seed: int, **kwargs) -> GraphBundle:
        """Load dataset with distributed-safe downloading.
        
        In multi-GPU settings, only rank 0 downloads and processes the raw data
        first. A barrier ensures all other ranks wait until the cache is ready
        before they load from the already-processed files.
        """
        if dist.is_initialized() and self.world_size > 1:
            if self.is_main:
                data = dataset.load(data_dir=data_dir, seed=seed, **kwargs)
            dist.barrier()
            if not self.is_main:
                data = dataset.load(data_dir=data_dir, seed=seed, **kwargs)
        else:
            data = dataset.load(data_dir=data_dir, seed=seed, **kwargs)
        return data

    def _validation_score_from_aggregate(self, metrics: Dict[str, Dict[str, float]]) -> float:
        """Score used for best-checkpoint selection from aggregate eval metrics.

        Controlled by ``--checkpoint_metric``:
        - ``auto`` (default): mean of available MRR and accuracy (legacy).
        - ``accuracy`` / ``roc_auc`` / ``mrr``: that key alone; falls back to
          accuracy then any finite aggregate value if missing/NaN.
        """
        agg = metrics["aggregate"]
        metric = str(getattr(self.args, "checkpoint_metric", "auto") or "auto")

        def _finite(key: str) -> Optional[float]:
            if key not in agg:
                return None
            val = float(agg[key])
            if val != val:  # NaN
                return None
            return val

        if metric != "auto":
            chosen = _finite(metric)
            if chosen is not None:
                return chosen
            for fallback in ("accuracy", "mrr", "roc_auc"):
                if fallback == metric:
                    continue
                fb = _finite(fallback)
                if fb is not None:
                    return fb
            for v in agg.values():
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if fv == fv:
                    return fv
            return float("nan")

        parts: list[float] = []
        for key in ("mrr", "accuracy"):
            fv = _finite(key)
            if fv is not None:
                parts.append(fv)
        if parts:
            return sum(parts) / len(parts)
        for v in agg.values():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv == fv:
                return fv
        return float("nan")

    def _dataset_is_link_prediction(self, dataset_name: str) -> bool:
        """True when ``dataset_name`` is a link-prediction benchmark (no node labels)."""
        dataset_args = get_datasetargs(dataset_name)
        return not dataset_args["has_node_labels"]

    def _nc_proximity_kwargs(self, seed: Optional[int] = None) -> dict:
        """NC eval packed-ball batching flags.

        When batching is on and ``--nc_proximity_batch_seed`` is unset, tie the
        partition RNG to the experiment seed so ``--seeds 0 1 2`` averages over
        different cuts.
        """
        batching = bool(getattr(self.args, "nc_proximity_batching", True))
        batch_seed = getattr(self.args, "nc_proximity_batch_seed", None)
        if batching and batch_seed is None and seed is not None:
            batch_seed = int(seed)
        return {
            "nc_proximity_batching": batching,
            "nc_proximity_batch_max_radius": getattr(
                self.args, "nc_proximity_batch_max_radius", None
            ),
            "nc_proximity_batch_seed": batch_seed,
        }

    def _eval_multigraph_loader_kwargs(
        self,
        seed: int,
        *,
        max_eval_samples,
        nc_proximity_batching: Optional[bool] = None,
    ) -> dict:
        """Shared MultiGraphLoader kwargs for eval (not training)."""
        kwargs = {
            "batch_size_node": effective_eval_batch_size_node(self.args),
            "batch_size_link": effective_eval_batch_size_link(self.args),
            "shuffle": False,
            "training": False,
            "rank": self.rank,
            "world_size": self.world_size,
            "max_eval_samples": max_eval_samples,
            "use_direction_head_queries": self._use_direction_head_queries(),
            "evaluate_head_predictions": self._evaluate_head_predictions(),
            "eval_subsample_seed": seed,
            "link_pred_eval": self._link_pred_eval(),
            "recall_k": self._recall_k(),
            "sampled_recall_num_neg": int(getattr(self.args, "sampled_recall_num_neg", 100)),
            "sampled_recall_max_sources": getattr(self.args, "sampled_recall_max_sources", None),
            **self._nc_proximity_kwargs(seed),
        }
        if nc_proximity_batching is not None:
            kwargs["nc_proximity_batching"] = nc_proximity_batching
        return kwargs

    def _preprocessing_kwargs(self) -> dict:
        return {
            "pca_before_normalization": getattr(
                self.args, "pca_before_normalization", False
            ),
            "graphland_different_transform": getattr(
                self.args, "graphland_different_transform", True
            ),
            "graphland_categorical_as_ordinals": getattr(
                self.args, "graphland_categorical_as_ordinals", True
            ),
            "drop_constant_train_features": getattr(
                self.args, "drop_constant_train_features", True
            ),
            "final_inductive_zscore": getattr(
                self.args, "final_inductive_zscore", True
            ),
            "drop_feature_indices": getattr(self.args, "drop_feature_indices", None),
            "row_norm_mode": getattr(self.args, "row_norm_mode", "l2"),
        }

    def _dataset_kwargs(
        self,
        *,
        dataset_name: Optional[str] = None,
        pca_target_dim: Optional[int] = None,
    ) -> dict:
        ignore_set = {
            n.upper()
            for n in (getattr(self.args, "ignore_features_datasets", None) or [])
        }
        ignore = bool(self.args.ignore_features)
        if dataset_name is not None and dataset_name.upper() in ignore_set:
            ignore = True
        if pca_target_dim is None:
            pca_dim = self.args.pca_target_dim
            per_ds = getattr(self.args, "pca_target_dim_datasets", None) or {}
            if dataset_name is not None and dataset_name.upper() in per_ds:
                pca_dim = per_ds[dataset_name.upper()]
            else:
                link_dim = getattr(self.args, "pca_target_dim_link", None)
                if (
                    link_dim is not None
                    and dataset_name is not None
                    and self._dataset_is_link_prediction(dataset_name)
                ):
                    pca_dim = link_dim
        else:
            pca_dim = pca_target_dim
        return {
            "ignore_features": ignore,
            "ignore_edge_types": getattr(self.args, "ignore_edge_types", False),
            "dummy_features": self.args.dummy_features,
            "pca_target_dim": pca_dim,
            "adaptive_pca_node_threshold": getattr(
                self.args, "adaptive_pca_node_threshold", 0
            ),
            "add_inverse_edges_kgs": getattr(self.args, "add_inverse_edges_KGs", True),
            "row_wise_norming": getattr(self.args, "row_wise_norming", False),
            "row_norm_mode": getattr(self.args, "row_norm_mode", "l2"),
            "nc_split_index": getattr(self.args, "nc_split_index", None),
            **self._preprocessing_kwargs(),
        }

    @staticmethod
    def _strip_wander_data_caches(data: Data) -> Data:
        """Drop walk caches so they are not cloned onto GPU with the graph."""
        for attr in (
            "_wander_walk_cache",
        ):
            if hasattr(data, attr):
                delattr(data, attr)
        return data

    def _copy_data_to_device(self, data: Data) -> Data:
        """Copy ``data`` to ``self.device`` without mutating the source object.

        Node-classification loaders yield the canonical ``loader.data``. PyG's
        ``Data.to(device)`` rebinds that object's tensors in place, so every
        visited train graph would stay GPU-resident for the rest of the run.
        """
        moved = self._strip_wander_data_caches(data.clone())
        return moved.to(self.device)

    def _move_batch_data(self, data: Data) -> Data:
        """Copy a training batch ``Data`` to device."""
        return self._copy_data_to_device(data)

    def _compute_step_loss(
        self,
        batch: Tensor,
        data: Data,
        model: nn.Module,
    ) -> Tensor:
        """Task loss for one (batch, data) item. Does not call
        ``loss.backward()`` or ``optimizer.step()``.
        """
        if batch.ndim == 1:
            # Native node classification: synthetic graphs supervise all
            # surviving test_mask nodes; real-world graphs supervise the loader
            # batch plus optional extra hidden train nodes that survive RW prune
            # (--nc_train_query_frac / --nc_train_query_cap).
            logits, targets = model(batch, data)
            return compute_node_classification_loss(logits, targets)

        if batch.shape[1] >= 4:
            predict_head = batch[:, 3].bool()
        else:
            predict_head = torch.zeros(batch.shape[0], dtype=torch.bool, device=batch.device)
        true_entities = torch.where(predict_head, batch[:, 0], batch[:, 2])
        negative_indices = sample_negative_entities(
            batch=batch,
            data=data,
            num_negatives=self.args.num_negatives,
            device=self.device,
        )

        candidate_entities = torch.cat([true_entities.unsqueeze(1), negative_indices], dim=1)
        scores = model(batch, data, candidate_entities)  # [B, 1+num_neg]
        return compute_link_prediction_loss(
            scores=scores,
            adversarial_temperature=self.args.adversarial_temperature,
            info_nce_temperature=self.args.info_nce_temperature,
        )

    def _log_nonfinite_loss(
        self,
        loss_value: float,
        *,
        dataset_name: str,
        epoch: int,
        step_idx: Optional[int],
        data: Data,
    ) -> None:
        """Log feature finiteness stats when a training loss is non-finite."""
        x = getattr(data, "x", None)
        x_stats: Dict[str, Any] = {}
        if isinstance(x, Tensor) and x.numel():
            x_stats = {
                "x_shape": tuple(x.shape),
                "x_has_nan": bool(torch.isnan(x).any().item()),
                "x_has_inf": bool(torch.isinf(x).any().item()),
                "x_min": float(x.min().item()),
                "x_max": float(x.max().item()),
            }
        print(
            f"[nan_loss] epoch={epoch} step={step_idx} dataset={dataset_name} "
            f"loss={loss_value} {x_stats}",
            flush=True,
        )

    def _check_model_finite(
        self,
        model: nn.Module,
        *,
        epoch: int,
        step_idx: Optional[int],
    ) -> None:
        """Abort if any parameter is non-finite after ``optimizer.step()``."""
        raw_model = model.module if isinstance(model, DDP) else model
        bad = [
            name
            for name, p in raw_model.named_parameters()
            if not torch.isfinite(p).all()
        ]
        if bad:
            raise RuntimeError(
                f"[nan_weights] epoch={epoch} step={step_idx} "
                f"non-finite parameters detected right after optimizer.step(): {bad}"
            )

    def _all_ranks_finite(self, loss: Tensor) -> bool:
        """Whether ``loss`` is finite on every rank (collective skip for DDP)."""
        finite = torch.isfinite(loss).detach().reshape(()).to(dtype=torch.uint8)
        if self.world_size > 1 and dist.is_available() and dist.is_initialized():
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        return bool(finite.item())

    def _train_step_multi(
        self,
        items: list,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int = 0,
        *,
        step_idx: Optional[int] = None,
    ) -> list:
        """Execute a single training step across multiple (graph, batch) items.

        Uses gradient accumulation: each item's loss is scaled by ``1/G`` and
        backpropagated immediately, accumulating into ``.grad``. A single
        ``optimizer.step()`` runs at the end. This is mathematically equivalent
        to backpropagating a summed-and-mean loss.

        For DDP we wrap all but the last backward in ``model.no_sync()`` so
        gradients are all-reduced only once per optimizer step.

        Returns
        -------
        list[tuple[str, float]]
            Per-graph ``(name, loss)`` pairs in input order, so the caller can
            bucket per-dataset losses correctly when a list mixes datasets.
        """
        model.train()
        optimizer.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        n = len(items)
        per_graph: list = []
        if n == 0:
            return per_graph

        is_ddp = isinstance(model, DDP)
        n_backward = 0
        for i, (name, batch, data) in enumerate(items):
            batch = batch.to(self.device)
            data = self._move_batch_data(data)
            sync_last_graph = (i == n - 1)
            sync_now = sync_last_graph or not is_ddp
            sync_ctx = nullcontext() if sync_now else model.no_sync()
            with sync_ctx:
                with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype):
                    loss = self._compute_step_loss(batch, data, model)
                loss_value = loss.item()
                per_graph.append((name, loss_value))
                if not self._all_ranks_finite(loss):
                    if not torch.isfinite(loss):
                        self._log_nonfinite_loss(
                            loss_value,
                            dataset_name=name,
                            epoch=epoch,
                            step_idx=step_idx,
                            data=data,
                        )
                    else:
                        print(
                            f"[nan_loss] epoch={epoch} step={step_idx} skipping "
                            f"item {name} (non-finite loss on a peer rank)",
                            flush=True,
                        )
                    continue
                (loss / n).backward()
                n_backward += 1

        if n == 1 and n_backward == 0:
            return per_graph

        optimizer.step()
        self._check_model_finite(model, epoch=epoch, step_idx=step_idx)

        return per_graph
    
    def train_epoch(
        self,
        loader: Union[MultiGraphLoader, "SyntheticGraphLoader"],
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int = 0,
    ) -> Dict[str, Dict[str, float]]:
        """Train for one epoch across all graphs.
        
        Args:
            loader: MultiGraphLoader or SyntheticGraphLoader for training data.
            model: The model.
            optimizer: Optimizer.
            epoch: Current epoch number (used to seed deterministic dataset order for DDP).
            
        Returns:
            Per-graph and aggregate loss metrics.
        """
        loader.set_epoch(epoch)
        per_graph_losses = defaultdict(list)
        epoch_batches = int(len(loader)) if hasattr(loader, "__len__") else int(
            getattr(self.args, "batch_per_epoch", 0) or 0
        )
        log_epoch_progress = self.is_main and epoch_batches > 2000
        log_cuda_mem = os.environ.get("LOG_REAL_TRAIN_GRAPHS", "0") == "1"

        for step_idx, item in enumerate(loader):
            step_items = item if isinstance(item, list) else [item]
            for name, loss in self._train_step_multi(
                step_items, model, optimizer, epoch, step_idx=step_idx,
            ):
                per_graph_losses[name].append(loss)
            # Drop step-local graph tensors before the next real/synth batch.
            del item
            if log_cuda_mem and torch.cuda.is_available():
                dev = torch.device(self.device)
                alloc_gb = torch.cuda.memory_allocated(dev) / 1e9
                reserved_gb = torch.cuda.memory_reserved(dev) / 1e9
                peak_gb = torch.cuda.max_memory_allocated(dev) / 1e9
                print(
                    f"[cuda_mem] epoch={epoch} step={step_idx} rank={self.rank} "
                    f"allocated_gb={alloc_gb:.2f} reserved_gb={reserved_gb:.2f} "
                    f"peak_gb={peak_gb:.2f}",
                    flush=True,
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if log_epoch_progress:
                done = step_idx + 1
                if done % 2000 == 0:
                    print(
                        f"[train] epoch={epoch} batches={done}/{epoch_batches}",
                        flush=True,
                    )

        keep_workers = bool(getattr(self.args, "synthetic_prefetch_keep_workers", True))
        close_synthetic_prefetch(loader, kill_workers=not keep_workers)
        
        # Compute per-graph average losses over the *finite* steps only. A
        # single non-finite step (skipped by the guard in _train_step_multi, so it
        # never reached the weights) would otherwise turn the whole epoch's
        # reported loss into nan and hide the real trajectory. The skipped
        # count is reported alongside so it stays visible instead of silent.
        per_graph_metrics = {}
        for name, losses in per_graph_losses.items():
            finite = [x for x in losses if math.isfinite(x)]
            n_skipped = len(losses) - len(finite)
            per_graph_metrics[name] = {
                "loss": (sum(finite) / len(finite)) if finite else float("nan"),
                "loss_nonfinite_steps": float(n_skipped),
            }
            if n_skipped and self.is_main:
                print(
                    f"[train] epoch={epoch} {name}: {n_skipped}/{len(losses)} steps "
                    f"had non-finite loss and were excluded from the epoch mean",
                    flush=True,
                )
        
        # Add aggregate metrics
        per_graph_metrics["aggregate"] = aggregate_metrics(per_graph_metrics)
        
        return per_graph_metrics

    @torch.no_grad()
    def _evaluate_recall_at_k(
        self,
        query_loader: QueryLoaderRecallAtK,
        bundle: GraphBundle,
        model: nn.Module,
        split: str,
    ) -> Dict[str, float]:
        """AnyGraph-style node-based Recall@K / NDCG@K.

        For each source node, score all nodes, mask train positives (and, for
        bipartite graphs, non-item candidates), take the top-K, and compute
        recall/NDCG against the source's true test tails. Metrics are
        macro-averaged over source nodes (reduced across DDP ranks).
        """
        model.eval()
        recall_k = int(getattr(self.args, "recall_k", 20))

        filter_index = get_train_filter_index(bundle, split)  # type: ignore[arg-type]

        is_bipartite = query_loader.is_bipartite
        candidate_offset = query_loader.candidate_offset

        recall_sum = 0.0
        ndcg_sum = 0.0
        source_count = 0

        for batch, data in query_loader:
            batch = batch.to(self.device)
            data = data.to(self.device)
            with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype):
                scores = model(batch, data)  # [B, num_nodes]
            scores_cpu = scores.float().cpu()
            del scores
            heads = batch[:, 0].cpu()

            for i in range(scores_cpu.shape[0]):
                src = int(heads[i])
                test_tails = query_loader.test_targets.get(src, [])
                n_test = len(test_tails)
                if n_test == 0:
                    continue

                scores_i = scores_cpu[i]
                if filter_index is not None:
                    # Homogeneous / default recall path uses relation id 0.
                    filt = filter_index.tails.get((src, 0))
                    if filt is not None and filt.numel() > 0:
                        scores_i[filt] = float("-inf")
                if is_bipartite and candidate_offset > 0:
                    scores_i[:candidate_offset] = float("-inf")

                k = min(recall_k, scores_i.shape[0])
                top_locs = torch.topk(scores_i, k).indices.tolist()
                top_rank = {t: r for r, t in enumerate(top_locs)}

                hits = 0
                dcg = 0.0
                for t in test_tails:
                    r = top_rank.get(t)
                    if r is not None:
                        hits += 1
                        dcg += 1.0 / math.log2(r + 2)
                max_dcg = sum(
                    1.0 / math.log2(r + 2) for r in range(min(n_test, recall_k))
                )
                recall_sum += hits / n_test
                ndcg_sum += (dcg / max_dcg) if max_dcg > 0 else 0.0
                source_count += 1

        if self.world_size > 1:
            stats = torch.tensor(
                [recall_sum, ndcg_sum, float(source_count)],
                dtype=torch.float64, device=self.device,
            )
            dist.all_reduce(stats)
            recall_sum = stats[0].item()
            ndcg_sum = stats[1].item()
            source_count = int(stats[2].item())

        denom = source_count if source_count > 0 else 1
        return {
            f"recall@{recall_k}": recall_sum / denom,
            f"ndcg@{recall_k}": ndcg_sum / denom,
        }

    @torch.no_grad()
    def _evaluate_sampled_recall(
        self,
        query_loader: QueryLoaderSampledRecall,
        model: nn.Module,
        split: str,
        *,
        dataset_name: str = "",
    ) -> Dict[str, float]:
        """Per-source Recall@K / NDCG@K on a cached pos ∪ sampled-neg list."""
        from data.sampled_recall import source_recall_ndcg

        model.eval()
        ks = (20, 50)
        sums = {f"recall@{k}": 0.0 for k in ks}
        sums.update({f"ndcg@{k}": 0.0 for k in ks})
        source_count = 0
        items_done = 0
        batch_idx = 0
        progress_total = int(query_loader.num_queries) if self.is_main else 0

        for batch, data in query_loader:
            rows = query_loader._iter_candidates
            if rows is None:
                raise RuntimeError("sampled-recall loader did not set _iter_candidates")
            n_cands = [int(pos.numel() + neg.numel()) for _, pos, neg in rows]
            cmax = max(n_cands) if n_cands else 1
            cand = torch.zeros(len(rows), cmax, dtype=torch.long)
            for i, (_, pos, neg) in enumerate(rows):
                tails = torch.cat([pos.long(), neg.long()], dim=0)
                cand[i, : tails.numel()] = tails
            batch = batch.to(self.device)
            data = data.to(self.device)
            cand = cand.to(self.device)
            with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                scores = model(batch, data, candidate_tails=cand)
            scores_cpu = scores.detach().float().cpu()
            del scores
            for i, (_, pos, neg) in enumerate(rows):
                n_pos = int(pos.numel())
                n_neg = int(neg.numel())
                if n_pos == 0:
                    continue
                row = source_recall_ndcg(
                    scores_cpu[i, :n_pos],
                    scores_cpu[i, n_pos : n_pos + n_neg],
                    ks=ks,
                )
                for key, val in row.items():
                    sums[key] += val
                source_count += 1
            items_done += int(batch.size(0))
            if self.is_main and progress_total > 0:
                _log_eval_progress(
                    dataset_name=dataset_name or split,
                    split=split,
                    is_native_node_cls=False,
                    batch_idx=batch_idx,
                    items_done=items_done,
                    total_items=progress_total,
                    correct_count=0,
                    total_count=0,
                    reciprocal_rank_sum=0.0,
                    rank_count=0,
                )
            batch_idx += 1

        if self.world_size > 1:
            keys = list(sums)
            vec = torch.tensor(
                [sums[k] for k in keys] + [float(source_count)],
                dtype=torch.float64,
                device=self.device,
            )
            dist.all_reduce(vec)
            for i, key in enumerate(keys):
                sums[key] = float(vec[i].item())
            source_count = int(vec[-1].item())

        denom = source_count if source_count > 0 else 1
        out = {key: val / denom for key, val in sums.items()}
        out["num_sources"] = float(source_count)
        return out

    @torch.no_grad()
    def evaluate(
        self,
        loader: MultiGraphLoader,
        model: nn.Module,
        split: str,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate on validation or test set.
        
        Supports two modes:
        - Link prediction: MRR, Hits@K
        - Native node classification: accuracy via argmax of logits; also
          reports ``roc_auc`` — binary AUROC (class-1 probability) when ``C=2``,
          else macro-averaged one-vs-rest AUROC over class probabilities.
        
        Args:
            loader: MultiGraphLoader for evaluation data (queries sharded by rank).
            model: The model (unwrapped, not DDP).
            split: "val" or "test".
            
        Returns:
            Per-graph and aggregate metrics (identical on all ranks after reduction).
        """
        model.eval()
        per_graph_metrics = {}
        
        for name, query_loader in loader.loaders.items():
            bundle = loader.bundles[name]
            dataset_data = loader.datasets[name]
            print("Evaluating on dataset: ", name, flush=True)

            from .query_loader import BatchLoaderNodeClassification
            is_native_node_cls = isinstance(query_loader, BatchLoaderNodeClassification)

            if isinstance(query_loader, QueryLoaderRecallAtK):
                per_graph_metrics[name] = self._evaluate_recall_at_k(
                    query_loader, bundle, model, split,
                )
                continue

            if isinstance(query_loader, QueryLoaderSampledRecall):
                per_graph_metrics[name] = self._evaluate_sampled_recall(
                    query_loader, model, split, dataset_name=name,
                )
                continue

            progress_total = (
                _eval_progress_total(
                    query_loader,
                    is_native_node_cls=is_native_node_cls,
                )
                if self.is_main
                else 0
            )
            items_done = 0
            batch_idx = 0
            
            correct_count = 0
            total_count = 0
            reciprocal_rank_sum = 0.0
            rank_count = 0
            hits1_count = 0
            hits3_count = 0
            hits10_count = 0
            lp_rank_accum = RankAccum() if not is_native_node_cls else None
            lp_rel_names = (
                relation_names_from_bundle(bundle) if not is_native_node_cls else None
            )
            cls_targets_chunks: list[Tensor] = []
            cls_proba_chunks: list[Tensor] = []
            
            filter_index = None
            use_head_filter = False
            if not is_native_node_cls:
                if self.is_main:
                    print(f"[lp filter] building/caching index for {name} ({split})...", flush=True)
                filter_index = get_eval_filter_index(bundle, split)  # type: ignore[arg-type]
                use_head_filter = self._use_direction_head_queries()
                if self.is_main:
                    print(
                        f"[lp filter] {name}: {filter_index.num_triples} triples, "
                        f"{len(filter_index.tails)} (h,r) keys",
                        flush=True,
                    )

            coverage_sums = defaultdict(float)
            coverage_count = 0

            # Cached train-KV eval (Phase A): precompute per-pass train-node
            # states once per dataset and attach them to the model; Phase B
            # (the batch loop below) then uses them as global-attention K/V.
            train_kv_cache = None
            if (
                is_native_node_cls
                and getattr(model, "cached_train_kv", False)
                and hasattr(model, "precompute_train_kv_cache")
            ):
                dataset_data = dataset_data.to(self.device)
                with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype):
                    train_kv_cache = model.precompute_train_kv_cache(dataset_data)
                model.attach_train_kv_cache(train_kv_cache)
                if self.is_main and self.world_size > 1:
                    print(
                        f"[eval] {name}: Phase B sharded across {self.world_size} GPUs "
                        f"({len(query_loader)} batches on rank 0).",
                        flush=True,
                    )
            elif self.is_main and self.world_size > 1 and is_native_node_cls:
                print(
                    f"[eval] {name}: eval batches sharded across {self.world_size} GPUs "
                    f"({len(query_loader)} batches on rank 0).",
                    flush=True,
                )

            resume_enabled = self._eval_batch_resume_enabled() and is_native_node_cls
            resume_path = ""
            resume_fp: Optional[Dict[str, Any]] = None
            resume_from = 0
            resume_batch_sizes: list[int] = []
            resume_node_id_chunks: list[Tensor] = []
            if resume_enabled:
                seed = int(getattr(self, "_current_seed", 0) or 0)
                resume_path = eval_batch_resume_path(
                    self.checkpoint_dir,
                    dataset_name=name,
                    split=split,
                    seed=seed,
                    rank=self.rank,
                )
                resume_fp = build_eval_batch_fingerprint(
                    self.args,
                    dataset_name=name,
                    split=split,
                    seed=seed,
                    rank=self.rank,
                    world_size=self.world_size,
                    n_eval_items=self._eval_resume_n_items(query_loader),
                )
                resume_from, loaded_resume = resolve_and_validate_resume(
                    path=resume_path,
                    fingerprint=resume_fp,
                    query_loader=query_loader,
                    world_size=self.world_size,
                    device=self.device,
                )
                if loaded_resume is not None and resume_from > 0:
                    restored = apply_resume_payload(loaded_resume)
                    items_done = restored["items_done"]
                    correct_count = restored["correct_count"]
                    total_count = restored["total_count"]
                    resume_batch_sizes = restored["batch_sizes"]
                    cls_targets_chunks = restored["cls_targets_chunks"]
                    cls_proba_chunks = restored["cls_proba_chunks"]
                    resume_node_id_chunks = restored["node_id_chunks"]
                    for cov_key, cov_val in restored["coverage_sums"].items():
                        coverage_sums[cov_key] += cov_val
                    coverage_count = restored["coverage_count"]
                    if self.is_main:
                        print(
                            f"[eval resume] {name} ({split}): skipping "
                            f"{resume_from} finished batches "
                            f"({items_done} items already scored) "
                            f"from {resume_path}",
                            flush=True,
                        )
                elif self.is_main:
                    print(
                        f"[eval resume] {name} ({split}): writing progress to "
                        f"{resume_path} (reuse this run dir to continue after preemption)",
                        flush=True,
                    )

            def _flush_nc_eval_resume() -> None:
                if not resume_enabled or not resume_path or resume_fp is None:
                    return
                save_eval_batch_resume(
                    resume_path,
                    fingerprint=resume_fp,
                    next_batch_idx=batch_idx,
                    items_done=items_done,
                    correct_count=correct_count,
                    total_count=total_count,
                    batch_sizes=resume_batch_sizes,
                    node_ids=cat_chunks(resume_node_id_chunks),
                    cls_targets=cat_chunks(cls_targets_chunks),
                    cls_probas=cat_chunks(cls_proba_chunks),
                    coverage_sums=dict(coverage_sums),
                    coverage_count=coverage_count,
                )

            with eval_preempt_flag() as preempt_flag:
              for batch, data in query_loader:
                if resume_enabled and batch_idx < resume_from:
                    batch_idx += 1
                    continue
                if batch.numel() == 0:
                    if resume_enabled:
                        resume_batch_sizes.append(0)
                    batch_idx += 1
                    if resume_enabled:
                        _flush_nc_eval_resume()
                        self._maybe_stop_eval_on_preempt(
                            preempt_flag,
                            dataset_name=name,
                            split=split,
                            resume_path=resume_path,
                        )
                    continue
                batch = batch.to(self.device)
                data = data.to(self.device)

                with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype):
                    scores = model(batch, data)

                scores_cpu = scores.cpu()
                del scores

                if hasattr(model, '_coverage_stats') and model._coverage_stats:
                    for cov_key, cov_val in model._coverage_stats.items():
                        coverage_sums[f"coverage/{cov_key}"] += cov_val
                    coverage_count += 1

                if is_native_node_cls:
                    # scores_cpu: [batch_size, num_classes]
                    # Targets come from the per-batch graph object: temporal
                    # snapshot datasets carry seed-time-specific labels on
                    # data.y (identical to dataset_data.y for static loaders).
                    targets = data.y[batch].argmax(dim=1).cpu()
                    predicted = scores_cpu.argmax(dim=1)
                    correct_count += (predicted == targets).sum().item()
                    total_count += targets.shape[0]
                    if scores_cpu.shape[1] >= 2:
                        cls_proba_chunks.append(
                            torch.softmax(scores_cpu.float(), dim=1)
                        )
                        cls_targets_chunks.append(targets)
                    items_done += int(batch.shape[0])
                else:
                    heads = batch[:, 0].cpu()
                    relations = batch[:, 1].cpu()
                    tails = batch[:, 2].cpu()
                    if batch.shape[1] >= 4:
                        predict_head = batch[:, 3].cpu().bool()
                    else:
                        predict_head = torch.zeros(batch.shape[0], dtype=torch.bool)
                    for i in range(scores_cpu.shape[0]):
                        scores_i = scores_cpu[i].clone()
                        if predict_head[i]:
                            true_entity = heads[i].item()
                            t = tails[i].item()
                            r = relations[i].item()
                            filt = (
                                filter_index.heads.get((t, r))
                                if use_head_filter and filter_index is not None
                                else None
                            )
                        else:
                            true_entity = tails[i].item()
                            h = heads[i].item()
                            r = relations[i].item()
                            filt = (
                                filter_index.tails.get((h, r))
                                if filter_index is not None
                                else None
                            )
                        true_score = apply_filtered_entity_mask_(
                            scores_i, filt, int(true_entity)
                        )

                        rank = int((scores_i >= true_score).sum().item())
                        reciprocal_rank_sum += 1.0 / rank if rank > 0 else 0.0
                        rank_count += 1
                        hits1_count += int(rank <= 1)
                        hits3_count += int(rank <= 3)
                        hits10_count += int(rank <= 10)
                        if lp_rank_accum is not None:
                            lp_rank_accum.add(rank, rel_id=int(r))
                    items_done += int(batch.shape[0])

                if self.is_main and progress_total > 0:
                    items_shown = items_done
                    if self.world_size > 1:
                        items_shown = min(items_done * self.world_size, progress_total)
                    _log_eval_progress(
                        dataset_name=name,
                        split=split,
                        is_native_node_cls=is_native_node_cls,
                        batch_idx=batch_idx,
                        items_done=items_shown,
                        total_items=progress_total,
                        correct_count=correct_count,
                        total_count=total_count,
                        reciprocal_rank_sum=reciprocal_rank_sum,
                        rank_count=rank_count,
                    )
                if resume_enabled:
                    resume_batch_sizes.append(int(batch.numel()))
                    resume_node_id_chunks.append(batch.detach().cpu().view(-1))
                batch_idx += 1
                if resume_enabled:
                    _flush_nc_eval_resume()
                    self._maybe_stop_eval_on_preempt(
                        preempt_flag,
                        dataset_name=name,
                        split=split,
                        resume_path=resume_path,
                    )

            if train_kv_cache is not None:
                model.attach_train_kv_cache(None)
                train_kv_cache.release()
                train_kv_cache = None

            roc_auc_val: Optional[float] = None
            if is_native_node_cls:
                # Every rank must enter the same collectives. Proximity batching
                # with a large batch_size can give some ranks zero NC batches.
                if cls_targets_chunks:
                    y_true_local = torch.cat(cls_targets_chunks, dim=0).numpy()
                    y_prob_local = torch.cat(cls_proba_chunks, dim=0).numpy()
                else:
                    y_true_local = np.empty((0,), dtype=np.int64)
                    y_prob_local = np.empty((0, 0), dtype=np.float64)
                if self.world_size > 1:
                    gathered = [None] * self.world_size if self.is_main else None
                    dist.gather_object((y_true_local, y_prob_local), gathered, dst=0)
                    roc_tensor = torch.zeros(1, dtype=torch.float64, device=self.device)
                    if self.is_main and gathered is not None:
                        trues = [
                            g[0] for g in gathered
                            if g is not None and getattr(g[0], "size", 0) > 0
                        ]
                        probs = [
                            g[1] for g in gathered
                            if g is not None and getattr(g[0], "size", 0) > 0
                        ]
                        if trues:
                            y_true_all = np.concatenate(trues)
                            max_c = max(int(p.shape[1]) for p in probs)
                            aligned = []
                            for p in probs:
                                if p.shape[1] == max_c:
                                    aligned.append(p)
                                else:
                                    z = np.zeros((p.shape[0], max_c), dtype=p.dtype)
                                    z[:, : p.shape[1]] = p
                                    aligned.append(z)
                            y_prob_all = np.concatenate(aligned)
                            roc_tensor[0] = macro_ovr_roc_auc_score(
                                y_true_all, y_prob_all
                            )
                    dist.broadcast(roc_tensor, src=0)
                    roc_auc_val = float(roc_tensor.item())
                elif y_true_local.size > 0:
                    roc_auc_val = macro_ovr_roc_auc_score(y_true_local, y_prob_local)
            
            if self.world_size > 1:
                if is_native_node_cls:
                    stats = torch.tensor(
                        [float(correct_count), float(total_count)],
                        dtype=torch.float64, device=self.device,
                    )
                else:
                    stats = torch.tensor(
                        [reciprocal_rank_sum, float(rank_count),
                         float(hits1_count), float(hits3_count), float(hits10_count)],
                        dtype=torch.float64, device=self.device,
                    )
                dist.all_reduce(stats)
                if is_native_node_cls:
                    correct_count, total_count = int(stats[0].item()), int(stats[1].item())
                else:
                    reciprocal_rank_sum = stats[0].item()
                    rank_count = int(stats[1].item())
                    hits1_count = int(stats[2].item())
                    hits3_count = int(stats[3].item())
                    hits10_count = int(stats[4].item())
                    if lp_rank_accum is not None:
                        n_rel = int(getattr(dataset_data, "num_relations", 0) or 0)
                        packed = lp_rank_accum.to_tensor(n_rel, self.device)
                        dist.all_reduce(packed)
                        lp_rank_accum = RankAccum.from_tensor(packed)
                        reciprocal_rank_sum = lp_rank_accum.rr
                        rank_count = lp_rank_accum.n
                        hits1_count = lp_rank_accum.h1
                        hits3_count = lp_rank_accum.h3
                        hits10_count = lp_rank_accum.h10
            
            if is_native_node_cls:
                metrics = {"accuracy": correct_count / total_count if total_count > 0 else 0.0}
                if roc_auc_val is not None:
                    metrics["roc_auc"] = roc_auc_val
            else:
                if lp_rank_accum is not None:
                    metrics = lp_rank_accum.as_metrics(lp_rel_names)
                else:
                    metrics = {
                        "mrr": reciprocal_rank_sum / rank_count if rank_count > 0 else 0.0,
                        "hits@1": hits1_count / rank_count if rank_count > 0 else 0.0,
                        "hits@3": hits3_count / rank_count if rank_count > 0 else 0.0,
                        "hits@10": hits10_count / rank_count if rank_count > 0 else 0.0,
                    }

            if self.is_main and progress_total > 0:
                if is_native_node_cls:
                    unit = "nodes"
                    done_items = total_count
                else:
                    unit = "queries"
                    done_items = rank_count
                msg = (
                    f"[eval progress] {name} ({split}): "
                    f"{done_items}/{progress_total} {unit} (0 remaining), "
                    f"batch {batch_idx} (done)"
                )
                if is_native_node_cls and total_count > 0:
                    msg += f", accuracy={correct_count / total_count:.4f}"
                elif rank_count > 0:
                    msg += f", mrr={reciprocal_rank_sum / rank_count:.4f}"
                print(msg, flush=True)

            if coverage_count > 0:
                for cov_key, cov_sum in coverage_sums.items():
                    metrics[cov_key] = cov_sum / coverage_count
            
            per_graph_metrics[name] = metrics

            if resume_enabled and resume_path:
                clear_eval_batch_resume(resume_path)
                if self.is_main:
                    print(
                        f"[eval resume] {name} ({split}): finished; "
                        f"cleared {resume_path}",
                        flush=True,
                    )

        
        per_graph_metrics["aggregate"] = aggregate_metrics(per_graph_metrics)
        return per_graph_metrics

    @torch.no_grad()
    def evaluate_synthetic(
        self,
        loader: "SyntheticGraphLoader",
        model: nn.Module,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate on synthetic graphs (native node classification).

        Unlike ``evaluate``, each batch carries its own freshly generated
        graph, so targets come from the per-batch ``data.y``. We only report
        accuracy: pooling predictions across batches would mix unrelated
        graphs, so a single ROC AUC would not be meaningful.
        """
        model.eval()
        correct_count = 0
        total_count = 0

        # Draw the same synthetic val tasks every epoch so resampling noise
        # does not swamp the epoch-to-epoch change in model quality.
        val_seed = int(getattr(self.args, "synthetic_val_seed", 12345))
        rng_state = _capture_rng_state()
        _seed_rngs(val_seed)
        try:
            for _name, batch, data in loader:
                if is_link_prediction(data) or getattr(data, "y", None) is None:
                    continue
                batch = batch.to(self.device)
                data = data.to(self.device)

                with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype):
                    scores = model(batch, data)

                predicted = scores.argmax(dim=1)
                targets = data.y[batch].argmax(dim=1)
                correct_count += int((predicted == targets).sum().item())
                total_count += int(targets.shape[0])
        finally:
            _restore_rng_state(rng_state)

        if self.world_size > 1:
            stats = torch.tensor(
                [float(correct_count), float(total_count)],
                dtype=torch.float64,
                device=self.device,
            )
            dist.all_reduce(stats)
            correct_count, total_count = int(stats[0].item()), int(stats[1].item())

        accuracy = correct_count / total_count if total_count > 0 else 0.0
        return {
            "synthetic": {"accuracy": accuracy},
            "aggregate": {"accuracy": accuracy},
        }

    def _format_eval_metric_summary(self, metrics: Dict[str, float]) -> str:
        skip = {"n_seeds", "n_splits"}
        parts = []
        for k, v in sorted(metrics.items()):
            if (
                k.startswith("coverage/")
                or k.startswith("num_queries")
                or k.endswith("_std")
                or k in skip
            ):
                continue
            std = metrics.get(f"{k}_std")
            if std is not None:
                parts.append(f"{k}={v:.4f} ± {float(std):.4f}")
            else:
                parts.append(f"{k}={v:.4f}")
        n_seeds = metrics.get("n_seeds")
        if n_seeds is not None and int(n_seeds) > 1:
            parts.append(f"n_seeds={int(n_seeds)}")
        n_splits = metrics.get("n_splits")
        if n_splits is not None and int(n_splits) > 1:
            parts.append(f"n_splits={int(n_splits)}")
        return ", ".join(parts)

    def _save_eval_metrics(
        self,
        seed: int,
        test_metrics: Dict[str, Dict[str, float]],
    ) -> Optional[str]:
        """Persist human-readable eval metrics alongside eval_outputs/*.pt."""
        if not test_metrics:
            return None

        metrics_path = os.path.join(self.checkpoint_dir, "metrics.json")
        payload = {
            "experiment_name": self.base_name,
            "seed": seed,
            "init_checkpoint": getattr(self.args, "init_checkpoint", None),
            "nc_split_index": getattr(self.args, "nc_split_index", None),
            "max_eval_samples_eval_only": getattr(
                self.args, "max_eval_samples_eval_only", None
            ),
            "metrics": {
                name: {k: float(v) for k, v in metric_dict.items()}
                for name, metric_dict in test_metrics.items()
            },
        }
        save_json(os.path.join(self.checkpoint_dir, f"metrics_seed{seed}.json"), payload)
        save_json(metrics_path, payload)

        if getattr(self.args, "eval_only", False):
            eval_args_path = save_eval_run_args(self.args.save_load_path, self.args)
            if self.is_main:
                print(f"Saved eval CLI args to {eval_args_path}")

        txt_path = os.path.join(self.checkpoint_dir, "metrics.txt")
        lines = []
        for ds_name in sorted(test_metrics):
            if ds_name == "aggregate":
                continue
            summary = self._format_eval_metric_summary(test_metrics[ds_name])
            lines.append(f"[eval_only] {ds_name} done: {summary}")
        if "aggregate" in test_metrics:
            summary = self._format_eval_metric_summary(test_metrics["aggregate"])
            lines.append(f"[eval_only] aggregate: {summary}")
        with open(txt_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
            handle.write("\n")

        return metrics_path

    def _apply_prior_complexity_update(
        self,
        *,
        epoch: int,
        prior_complexity: float,
        _build_synthetic_loaders,
        _build_real_train_loader,
        _mix: bool,
        _p_real: float,
        synth_train_loader,
        train_loader,
    ):
        """Apply async or inline prior-complexity increases by rebuilding loaders."""
        override = read_prior_complexity_override(self.checkpoint_dir, epoch)
        new_complexity = prior_complexity
        if override is not None and override > prior_complexity:
            new_complexity = override
        if new_complexity <= prior_complexity:
            return prior_complexity, synth_train_loader, train_loader

        close_synthetic_prefetch(train_loader, kill_workers=True)
        close_synthetic_prefetch(synth_train_loader, kill_workers=True)
        synth_train_loader = _build_synthetic_loaders(new_complexity)
        if _mix:
            real_train_loader = _build_real_train_loader()
            train_loader = MixedTrainLoader(
                real=real_train_loader,
                synthetic=synth_train_loader,
                p_real=_p_real,
                batch_per_epoch=self.args.batch_per_epoch,
            )
        else:
            train_loader = synth_train_loader
        if self.is_main:
            print(f"Prior complexity increased to {new_complexity:.2f}")
            if getattr(self.args, "skip_eval", False):
                wandb.log({
                    "train_epoch": epoch,
                    "prior_complexity_active": new_complexity,
                })
        return new_complexity, synth_train_loader, train_loader

    def _min_train_nodes(self, datasets_train: Dict[str, GraphBundle]) -> Optional[int]:
        min_train: Optional[int] = None
        for bundle in datasets_train.values():
            data = bundle.train if bundle is not None else None
            if data is None or not hasattr(data, "train_mask") or data.train_mask is None:
                continue
            n_train = int(data.train_mask.sum().item())
            min_train = n_train if min_train is None else min(min_train, n_train)
        return min_train

    def _should_run_train_eval(self, datasets_train: Dict[str, GraphBundle]) -> bool:
        if getattr(self.args, "skip_train_eval", True):
            return False
        min_train = self._min_train_nodes(datasets_train)
        if min_train is None:
            return True
        eval_bs = effective_eval_batch_size_node(self.args)
        if eval_bs >= min_train:
            if self.is_main:
                print(
                    f"Skipping train inline eval: eval_batch_size_node={eval_bs} "
                    f">= min train nodes={min_train}"
                )
            return False
        return True

    def _make_synthetic_val_loader(
        self,
        prior_complexity: float,
    ) -> "SyntheticGraphLoader":
        """Pinned synthetic-val loader for epoch eval."""
        overrides = prior_fixed_hp_overrides(self.args)
        prior = SCMPrior(
            classification_only=True,
            fixed_hp=get_default_fixed_hp(prior_complexity, **overrides),
            sampled_hp=get_default_sampled_hp(prior_complexity),
        )
        effective_val = getattr(self.args, "synthetic_val_steps", 200) // max(
            1, self.world_size
        )
        return SyntheticGraphLoader(
            prior=prior,
            batch_size=effective_eval_batch_size_node(self.args),
            batch_per_epoch=effective_val,
            pca_target_dim_node_syn=self.args.pca_target_dim_node_syn,
            pca_target_dim_link_syn=self.args.pca_target_dim_link_syn,
            adaptive_pca_node_threshold=getattr(
                self.args, "adaptive_pca_node_threshold", 0
            ),
            is_training=False,
            balanced_training=False,
            prefetch=getattr(self.args, "synthetic_prefetch", 2),
            prefetch_workers=getattr(self.args, "synthetic_prefetch_workers", 1),
            prefetch_mp=getattr(self.args, "synthetic_prefetch_mp", True),
            num_graphs_per_step=1,
            row_wise_norming=getattr(self.args, "row_wise_norming", False),
            dummy_features=self.args.dummy_features,
            ignore_features=self.args.ignore_features,
            link_pred_prob=getattr(self.args, "synthetic_link_pred_prob", 0.0),
            deterministic_seed=int(getattr(self.args, "synthetic_val_seed", 12345)),
            **self._preprocessing_kwargs(),
        )

    def _build_training_eval_loaders(
        self,
        datasets_train: Dict[str, GraphBundle],
        datasets_test: Dict[str, GraphBundle],
        seed: int,
        prior_complexity: float,
        eval_split: str = "val",
    ) -> Dict[str, Any]:
        """Build eval loaders used during training-time epoch evaluation."""
        if eval_split not in ("val", "test"):
            raise ValueError(f"eval_split must be 'val' or 'test', got {eval_split!r}")
        _synthetic = getattr(self.args, "synthetic_prior", False)

        val_loader = None
        if datasets_test:
            val_loader = MultiGraphLoader(
                bundles=datasets_test,
                split=eval_split,
                **self._eval_multigraph_loader_kwargs(
                    seed,
                    max_eval_samples=self.args.max_eval_samples_training,
                ),
            )

        val_loader_synthetic = None
        train_eval_loader = None
        if _synthetic:
            val_loader_synthetic = self._make_synthetic_val_loader(prior_complexity)
        else:
            if self._should_run_train_eval(datasets_train):
                # Eval-mode loader over the train split (not training=True): otherwise
                # NC ignores max_eval_samples and iterates the full train mask.
                #
                # Never use proximity batching here. BFS clusters on the train mask
                # are nearly class-pure, so masking a batch erases a local labeled
                # neighborhood. Val/test queries sit outside train_mask, so their
                # local train labels stay visible; train queries lose that context
                # and (especially with --ignore_features) rankings can invert.
                train_eval_loader = MultiGraphLoader(
                    bundles=datasets_train,
                    split="train",
                    **self._eval_multigraph_loader_kwargs(
                        seed,
                        max_eval_samples=self.args.max_eval_samples_training,
                        nc_proximity_batching=False,
                    ),
                )

        return {
            "synthetic": _synthetic,
            "val_loader": val_loader,
            "val_loader_synthetic": val_loader_synthetic,
            "train_eval_loader": train_eval_loader,
            "prior_complexity": prior_complexity,
            "eval_split": eval_split,
        }

    def run_training_epoch_eval(
        self,
        model: nn.Module,
        datasets_train: Dict[str, GraphBundle],
        datasets_test: Dict[str, GraphBundle],
        seed: int,
        prior_complexity: float,
        eval_split: str = "val",
    ) -> Dict[str, Any]:
        """Run the same per-epoch metrics as inline training eval (val split).
        """
        loaders = self._build_training_eval_loaders(
            datasets_train,
            datasets_test,
            seed,
            prior_complexity,
            eval_split=eval_split,
        )
        eval_model = model.module if isinstance(model, DDP) else model
        val_metrics = None
        if loaders["val_loader"] is not None:
            val_metrics = self.evaluate(loaders["val_loader"], eval_model, eval_split)
        val_synthetic_metrics = None
        train_eval_metrics = None
        if loaders["synthetic"]:
            val_synthetic_metrics = self.evaluate_synthetic(
                loaders["val_loader_synthetic"],
                eval_model,
            )
        else:
            if loaders["train_eval_loader"] is not None:
                train_eval_metrics = self.evaluate(
                    loaders["train_eval_loader"],
                    eval_model,
                    "train",
                )
        if val_metrics is not None:
            val_score = self._validation_score_from_aggregate(val_metrics)
        elif val_synthetic_metrics is not None:
            val_score = float(val_synthetic_metrics["aggregate"]["accuracy"])
        else:
            val_score = float("nan")

        return {
            "val_metrics": val_metrics,
            "val_synthetic_metrics": val_synthetic_metrics,
            "train_eval_metrics": train_eval_metrics,
            "test_metrics": None,
            "val_score": val_score,
            "prior_complexity": prior_complexity,
            "synthetic": loaders["synthetic"],
            "eval_split": eval_split,
        }

    @staticmethod
    def _eval_metrics_to_wandb_log_dict(
        epoch: int,
        *,
        train_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        val_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        val_synthetic_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        train_eval_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        test_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        prior_complexity: Optional[float] = None,
        synthetic: bool = False,
        eval_split: str = "val",
    ) -> Dict[str, Any]:
        log_dict: Dict[str, Any] = {"epoch": epoch}
        primary_prefix = eval_split if eval_split in ("val", "test") else "val"
        # Per-relation LP breakdowns (mrr/<rel>, hits@k/<rel>, num_queries/<rel>)
        # and aggregate num_queries stay in on-disk eval dumps; set
        # WANDB_LOG_PER_RELATION=1 to push per-relation keys (num_queries never).
        if train_metrics is not None:
            for name, metrics in train_metrics.items():
                for metric_name, value in metrics.items():
                    if not should_log_metric_to_wandb(metric_name):
                        continue
                    log_dict[f"train/{name}/{metric_name}"] = value
        if synthetic:
            if prior_complexity is not None:
                log_dict["prior_complexity"] = prior_complexity
            for prefix, source_metrics in [
                (primary_prefix, val_metrics),
                ("val_synthetic", val_synthetic_metrics),
            ]:
                if source_metrics is None:
                    continue
                for name, metrics in source_metrics.items():
                    for metric_name, value in metrics.items():
                        if metric_name.startswith("coverage/"):
                            cov_key = metric_name[len("coverage/") :]
                            log_dict[f"coverage/{prefix}/{name}/{cov_key}"] = value
                        elif should_log_metric_to_wandb(metric_name):
                            log_dict[f"{prefix}/{name}/{metric_name}"] = value
        else:
            # Primary split metrics live in ``val_metrics`` but are logged under
            # ``primary_prefix`` (val or test). Optional ``test_metrics`` is only
            # present when also evaluating test alongside val — skip it when the
            # primary split is already test to avoid a no-op / double-log.
            sources = [
                ("train_eval", train_eval_metrics),
                (primary_prefix, val_metrics),
            ]
            if test_metrics is not None and primary_prefix != "test":
                sources.append(("test", test_metrics))
            for prefix, source_metrics in sources:
                if source_metrics is None:
                    continue
                for name, metrics in source_metrics.items():
                    for metric_name, value in metrics.items():
                        if metric_name.startswith("coverage/"):
                            cov_key = metric_name[len("coverage/") :]
                            log_dict[f"coverage/{prefix}/{name}/{cov_key}"] = value
                        elif should_log_metric_to_wandb(metric_name):
                            log_dict[f"{prefix}/{name}/{metric_name}"] = value
        return log_dict

    def run_final_test_eval(
        self,
        model: nn.Module,
        datasets_test: Dict[str, GraphBundle],
        seed: int,
        best_model_state: Dict[str, Tensor],
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate the test split once using the provided model weights."""
        eval_model = model.module if isinstance(model, DDP) else model
        eval_model.load_state_dict(best_model_state)

        train_ts = int(self.args.wander_test_samples)
        final_ts_override = getattr(self.args, "final_test_wander_samples", None)
        eval_ts = train_ts if final_ts_override is None else int(final_ts_override)
        prev_ts = int(getattr(eval_model, "test_samples", train_ts))
        restored_ts = False
        if eval_ts != prev_ts:
            eval_model.test_samples = eval_ts
            restored_ts = True

        train_kv = int(getattr(eval_model, "train_kv_passes", self.args.wander_train_kv_passes))
        final_kv_override = getattr(self.args, "final_test_train_kv_passes", None)
        eval_kv = train_kv if final_kv_override is None else int(final_kv_override)
        prev_kv = int(getattr(eval_model, "train_kv_passes", train_kv))
        restored_kv = False
        if eval_kv != prev_kv:
            eval_model.train_kv_passes = eval_kv
            restored_kv = True

        if self.is_main and (restored_ts or restored_kv):
            bits = []
            if restored_ts:
                bits.append(
                    f"wander_test_samples={eval_ts} (inline training eval used {train_ts})"
                )
            if restored_kv:
                bits.append(
                    f"wander_train_kv_passes={eval_kv} "
                    f"(inline training eval used {int(self.args.wander_train_kv_passes)})"
                )
            print("Final test eval: " + "; ".join(bits))

        test_loader = MultiGraphLoader(
            bundles=datasets_test,
            split="test",
            **self._eval_multigraph_loader_kwargs(
                seed,
                max_eval_samples=None,
            ),
        )
        try:
            return self.evaluate(test_loader, eval_model, "test")
        finally:
            if restored_ts:
                eval_model.test_samples = prev_ts
            if restored_kv:
                eval_model.train_kv_passes = prev_kv

    def load_checkpoint_model_state(
        self,
        model: nn.Module,
        checkpoint_path: str,
    ) -> Dict[str, Any]:
        """Load model weights from an epoch checkpoint and return checkpoint metadata."""
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        raw_model = model.module if isinstance(model, DDP) else model
        raw_model.load_state_dict(filter_obsolete_checkpoint_keys(state_dict), strict=False)
        return ckpt

    def load_datasets(self, seed: int) -> tuple[Dict[str, GraphBundle], Dict[str, GraphBundle]]:
        """Load train and test dataset bundles for a seed."""
        datasets_train: Dict[str, GraphBundle] = {}
        if self.is_main:
            print("TRAINING DATASETS:")
        for dataset_name in self.args.train_datasets:
            dataset_args = get_datasetargs(dataset_name)
            dataset = DataSet(
                dataset_args, **self._dataset_kwargs(dataset_name=dataset_name)
            )
            bundle = self._load_dataset_safe(
                dataset,
                data_dir=self.args.data_dir,
                seed=seed,
            )
            datasets_train[dataset_name] = bundle
            if self.is_main:
                _print_bundle_stats(dataset_name, bundle)

        datasets_test: Dict[str, GraphBundle] = {}
        if self.is_main:
            print(f"{'-'*50}")
            print("TESTING DATASETS:")
        for dataset_name in self.args.test_datasets:
            if dataset_name in datasets_train:
                datasets_test[dataset_name] = datasets_train[dataset_name]
                if self.is_main:
                    _print_bundle_stats(dataset_name, datasets_train[dataset_name])
                continue
            if self.is_main:
                print(f"Loading {dataset_name}...", flush=True)
            dataset_args = get_datasetargs(dataset_name)
            dataset = DataSet(
                dataset_args, **self._dataset_kwargs(dataset_name=dataset_name)
            )
            try:
                bundle = self._load_dataset_safe(
                    dataset,
                    data_dir=self.args.data_dir,
                    seed=seed,
                )
            except Exception:
                if self.args.eval_only:
                    if self.is_main:
                        print(
                            f"{dataset_name}: SKIPPED (failed to load):\n"
                            f"{traceback.format_exc()}"
                        )
                    continue
                raise
            datasets_test[dataset_name] = bundle
            if self.is_main:
                _print_bundle_stats(dataset_name, bundle)
        return datasets_train, datasets_test
    
    def trainer(
        self,
        datasets_train: Dict[str, GraphBundle],
        datasets_test: Dict[str, GraphBundle],
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        seed: int,
        start_epoch: int = 0,
    ) -> Dict[str, any]:
        """Full training loop.
        
        Args:
            datasets_train: Dict mapping dataset names to Data objects.
            datasets_test: Dict mapping dataset names to Data objects.
            model: The model.
            optimizer: Optimizer.
            seed: Random seed for this run.
            start_epoch: Epoch to start from (0 for fresh, >0 for resume).
            
        Returns:
            Dict with best metrics and training history.
        """
        _synthetic = getattr(self.args, "synthetic_prior", False)
        _prefetch = getattr(self.args, "synthetic_prefetch", 2)
        _prefetch_workers = getattr(self.args, "synthetic_prefetch_workers", 1)
        _prefetch_mp = getattr(self.args, "synthetic_prefetch_mp", True)
        _num_graphs_batch = getattr(self.args, "num_graphs_batch", 1)
        _p_real = getattr(self.args, "real_world_graph_prob", 0.0)
        _mix = (
            _synthetic
            and _p_real > 0.0
            and len(getattr(self.args, "train_datasets", []) or []) > 0
        )
        balanced_node_training = not getattr(self.args, "node_cls_random_training_batches", True)

        if self.world_size > 1 and _num_graphs_batch % self.world_size != 0:
            raise ValueError(
                f"--num_graphs_batch ({_num_graphs_batch}) must be divisible by "
                f"world_size ({self.world_size}) when using multi-GPU training."
            )

        def _build_synthetic_loaders(complexity: float):
            prior = SCMPrior(
                classification_only=True,
                fixed_hp=get_default_fixed_hp(
                    complexity, **prior_fixed_hp_overrides(self.args)
                ),
                sampled_hp=get_default_sampled_hp(complexity),
            )
            return SyntheticGraphLoader(
                prior=prior,
                batch_size=self.args.batch_size_node,
                batch_per_epoch=self.args.batch_per_epoch,
                pca_target_dim_node_syn=self.args.pca_target_dim_node_syn,
                pca_target_dim_link_syn=self.args.pca_target_dim_link_syn,
                adaptive_pca_node_threshold=getattr(
                    self.args, "adaptive_pca_node_threshold", 0
                ),
                is_training=True,
                balanced_training=balanced_node_training,
                prefetch=_prefetch,
                prefetch_workers=_prefetch_workers,
                prefetch_mp=_prefetch_mp,
                num_graphs_per_step=_num_graphs_batch,
                ddp_world_size=self.world_size,
                row_wise_norming=getattr(self.args, "row_wise_norming", False),
                dummy_features=self.args.dummy_features,
                ignore_features=self.args.ignore_features,
                link_pred_prob=getattr(self.args, "synthetic_link_pred_prob", 0.0),
                batch_size_link=self.args.batch_size_link,
                **self._preprocessing_kwargs(),
            )

        def _build_real_train_loader() -> MultiGraphLoader:
            return MultiGraphLoader(
                bundles=datasets_train,
                split="train",
                batch_size_node=self.args.batch_size_node,
                batch_size_link=self.args.batch_size_link,
                shuffle=True,
                training=True,
                rank=self.rank,
                world_size=self.world_size,
                batch_per_epoch=self.args.batch_per_epoch,
                max_eval_samples=self.args.max_eval_samples_training,
                balanced_training=balanced_node_training,
                num_graphs_per_step=_num_graphs_batch,
                use_direction_head_queries=self._use_direction_head_queries(),
                eval_subsample_seed=seed,
                train_without_replacement=bool(
                    getattr(self.args, "train_without_replacement", False)
                ),
            )

        # Always defined so non-synthetic finetune/eval paths can pass it through.
        prior_complexity = getattr(self.args, "prior_complexity_start", 1.0)
        if _synthetic:
            if self.args.resume_checkpoint:
                ckpt = torch.load(self.args.resume_checkpoint, map_location="cpu")
                prior_complexity = ckpt.get("prior_complexity", prior_complexity)
            synth_train_loader = _build_synthetic_loaders(prior_complexity)
            if _mix:
                real_train_loader = _build_real_train_loader()
                train_loader = MixedTrainLoader(
                    real=real_train_loader,
                    synthetic=synth_train_loader,
                    p_real=_p_real,
                    batch_per_epoch=self.args.batch_per_epoch,
                )
            else:
                train_loader = synth_train_loader
        else:
            train_loader = _build_real_train_loader()
        best_model_state = None
        best_metrics = None
        best_val_score = None
        best_model_dirty = False
        skip_eval = getattr(self.args, "skip_eval", False)
        patience = int(getattr(self.args, "patience", 0) or 0)
        epochs_no_improve = 0
        last_completed_epoch = start_epoch - 1
        
        for epoch in range(start_epoch, self.args.max_epochs):
            if _synthetic and getattr(self.args, "prior_complexity_adapt", False):
                prior_complexity, synth_train_loader, train_loader = self._apply_prior_complexity_update(
                    epoch=epoch,
                    prior_complexity=prior_complexity,
                    _build_synthetic_loaders=_build_synthetic_loaders,
                    _build_real_train_loader=_build_real_train_loader,
                    _mix=_mix,
                    _p_real=_p_real,
                    synth_train_loader=synth_train_loader,
                    train_loader=train_loader,
                )

            train_metrics = self.train_epoch(train_loader, model, optimizer, epoch=epoch)
            eval_split = "val"

            if skip_eval:
                if self.is_main:
                    log_dict = self._eval_metrics_to_wandb_log_dict(
                        epoch,
                        train_metrics=train_metrics,
                        prior_complexity=prior_complexity if _synthetic else None,
                    )
                    log_dict.pop("epoch", None)
                    log_dict["train_epoch"] = epoch
                    if _synthetic:
                        log_dict["prior_complexity_active"] = prior_complexity
                    wandb.log(log_dict, step=epoch, commit=True)
                    if (epoch + 1) % 10 == 0 or epoch == 0:
                        agg_train_loss = train_metrics["aggregate"]["loss"]
                        print(f"Epoch {epoch}: train_loss={agg_train_loss:.4f}")
            else:
                eval_result = self.run_training_epoch_eval(
                    model,
                    datasets_train,
                    datasets_test,
                    seed,
                    prior_complexity,
                    eval_split=eval_split,
                )
                val_metrics = eval_result["val_metrics"]
                val_synthetic_metrics = eval_result["val_synthetic_metrics"]
                train_eval_metrics = eval_result["train_eval_metrics"]
                test_metrics = eval_result.get("test_metrics")
                val_score = eval_result["val_score"]

                if best_val_score is None or val_score > best_val_score:
                    best_val_score = val_score
                    model_to_save = model.module if isinstance(model, DDP) else model
                    best_model_state = deepcopy(model_to_save.state_dict())
                    best_metrics = {
                        "val": val_metrics,
                        "epoch": epoch,
                        "eval_split": eval_split,
                    }
                    if test_metrics is not None:
                        best_metrics["test"] = test_metrics
                    best_model_dirty = True
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if self.is_main:
                    log_dict = self._eval_metrics_to_wandb_log_dict(
                        epoch,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                        val_synthetic_metrics=val_synthetic_metrics,
                        train_eval_metrics=train_eval_metrics,
                        test_metrics=test_metrics,
                        prior_complexity=prior_complexity if _synthetic else None,
                        synthetic=_synthetic,
                        eval_split=eval_split,
                    )
                    # ``define_metric("test/*"|"val/*"|"train_eval/*", step_metric=eval_epoch)``
                    log_dict["eval_epoch"] = epoch
                    if patience > 0:
                        log_dict["patience/epochs_no_improve"] = epochs_no_improve
                        log_dict["patience/limit"] = patience
                    wandb.log(log_dict, step=epoch, commit=True)
                    save_epoch_eval_metrics(
                        self.checkpoint_dir,
                        epoch,
                        val_score=val_score,
                        val_metrics=val_metrics,
                        val_synthetic_metrics=val_synthetic_metrics,
                        train_eval_metrics=train_eval_metrics,
                        test_metrics=test_metrics,
                        prior_complexity=prior_complexity if _synthetic else None,
                        checkpoint_path=get_checkpoint_path(
                            self.args.save_load_path, self.base_name, seed, epoch=epoch
                        ),
                    )

                    if (epoch + 1) % 10 == 0 or epoch == 0:
                        agg_train_loss = train_metrics["aggregate"]["loss"]
                        score_name = f"{eval_split}_score"
                        if _synthetic:
                            syn_acc = val_synthetic_metrics["aggregate"]["accuracy"]
                            print(
                                f"Epoch {epoch}: train_loss={agg_train_loss:.4f}, "
                                f"{score_name}={val_score:.4f}, syn_acc={syn_acc:.4f}"
                                f", complexity={prior_complexity:.2f}"
                            )
                        else:
                            train_eval_part = ""
                            if train_eval_metrics is not None:
                                agg_train_eval = train_eval_metrics["aggregate"]
                                train_eval_score = next(iter(agg_train_eval.values()))
                                train_eval_part = f"train_eval={train_eval_score:.4f}, "
                            extra = ""
                            if test_metrics is not None:
                                ta = test_metrics["aggregate"]
                                t_auc = ta.get("roc_auc")
                                t_acc = ta.get("accuracy")
                                if t_auc is not None:
                                    extra = f", test_auc={float(t_auc):.4f}"
                                elif t_acc is not None:
                                    extra = f", test_acc={float(t_acc):.4f}"
                            print(
                                f"Epoch {epoch}: train_loss={agg_train_loss:.4f}, "
                                f"{train_eval_part}"
                                f"{score_name}={val_score:.4f}{extra}"
                            )

                if _synthetic and getattr(self.args, "prior_complexity_adapt", False):
                    syn_acc = val_synthetic_metrics["aggregate"]["accuracy"]
                    if syn_acc > 0.65 and prior_complexity < 1.0:
                        prior_complexity = min(1.0, prior_complexity + 0.05)
                        close_synthetic_prefetch(train_loader, kill_workers=True)
                        close_synthetic_prefetch(synth_train_loader, kill_workers=True)
                        synth_train_loader = _build_synthetic_loaders(prior_complexity)
                        if _mix:
                            real_train_loader = _build_real_train_loader()
                            train_loader = MixedTrainLoader(
                                real=real_train_loader,
                                synthetic=synth_train_loader,
                                p_real=_p_real,
                                batch_per_epoch=self.args.batch_per_epoch,
                            )
                        else:
                            train_loader = synth_train_loader
                        if self.is_main:
                            print(f"Prior complexity increased to {prior_complexity:.2f}")

            if self.args.save_checkpoint_interval > 0 and (epoch + 1) % self.args.save_checkpoint_interval == 0:
                model_to_save = model.module if isinstance(model, DDP) else model
                epoch_ckpt_path = get_checkpoint_path(self.args.save_load_path, self.base_name, seed, epoch=epoch)
                ckpt_dict = {
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                }
                if _synthetic:
                    ckpt_dict["prior_complexity"] = prior_complexity
                torch.save(ckpt_dict, epoch_ckpt_path)
                if self.is_main:
                    print(f"Saved epoch {epoch} checkpoint to {epoch_ckpt_path}")

            if self.is_main and not skip_eval and best_model_dirty and best_model_state is not None:
                best_ckpt_path = get_checkpoint_path(self.args.save_load_path, self.base_name, seed, epoch="best")
                torch.save(best_model_state, best_ckpt_path)
                print(f"Saved current best model (epoch {best_metrics['epoch']}) to {best_ckpt_path}")
                best_model_dirty = False

            last_completed_epoch = epoch
            if self.is_main:
                write_training_status(
                    self.checkpoint_dir,
                    status="running",
                    last_completed_epoch=epoch,
                    max_epochs=self.args.max_epochs,
                )

            if (
                not skip_eval
                and patience > 0
                and epochs_no_improve >= patience
            ):
                if self.is_main:
                    best_ep = best_metrics["epoch"] if best_metrics is not None else None
                    print(
                        f"Early stopping at epoch {epoch}: no {eval_split} improvement "
                        f"for {patience} epoch(s) "
                        f"(best={best_val_score:.4f} at epoch {best_ep})"
                    )
                    wandb.log(
                        {
                            "eval_epoch": epoch,
                            "patience/early_stopped": 1,
                            "patience/epochs_no_improve": epochs_no_improve,
                            "patience/limit": patience,
                        },
                        step=epoch,
                        commit=True,
                    )
                break
        
        close_synthetic_prefetch(train_loader, kill_workers=True)

        if self.is_main:
            write_training_status(
                self.checkpoint_dir,
                status="complete",
                last_completed_epoch=last_completed_epoch,
                max_epochs=self.args.max_epochs,
            )

        if skip_eval:
            return {
                "best_metrics": None,
                "best_model_state": None,
            }

        skip_final_test = bool(getattr(self.args, "skip_final_test", False))
        if datasets_test and not skip_final_test:
            test_metrics = self.run_final_test_eval(
                model,
                datasets_test,
                seed,
                best_model_state,
            )
            best_metrics["test"] = test_metrics
        else:
            test_metrics = None
            if self.is_main and skip_final_test:
                print("Skipping final test eval (--skip_final_test); best checkpoint still saved")

        if self.is_main:
            checkpoint_path = get_checkpoint_path(self.args.save_load_path, self.base_name, seed, epoch="best")
            torch.save(best_model_state, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
            if best_metrics is not None and best_val_score is not None:
                save_best_val_metrics(
                    self.checkpoint_dir,
                    best_epoch=int(best_metrics["epoch"]),
                    best_val_score=float(best_val_score),
                    val_metrics=best_metrics.get("val"),
                    extra={
                        "checkpoint_metric": getattr(
                            self.args, "checkpoint_metric", "auto"
                        ),
                        "eval_split": best_metrics.get("eval_split", "val"),
                    },
                )
            final_extra: Dict[str, Any] = {}
            final_ts_override = getattr(self.args, "final_test_wander_samples", None)
            if final_ts_override is not None:
                final_extra["wander_test_samples"] = int(final_ts_override)
                final_extra["training_wander_test_samples"] = int(
                    self.args.wander_test_samples
                )
            final_kv_override = getattr(self.args, "final_test_train_kv_passes", None)
            if final_kv_override is not None:
                final_extra["wander_train_kv_passes"] = int(final_kv_override)
                final_extra["training_wander_train_kv_passes"] = int(
                    self.args.wander_train_kv_passes
                )
            if test_metrics is not None:
                save_final_test_metrics(
                    self.checkpoint_dir,
                    best_epoch=best_metrics["epoch"],
                    test_metrics=test_metrics,
                    extra=final_extra or None,
                )
        
        return {
            "best_metrics": best_metrics,
            "best_model_state": best_model_state,
        }
    
    def run_single_seed(
        self,
        datasets_train: Dict[str, GraphBundle],
        datasets_test: Dict[str, Optional[GraphBundle]],
        seed: int,
    ) -> Dict[str, any]:
        """Run experiment for a single seed.
        
        Args:
            datasets_train: Dict mapping dataset names to Data objects.
            datasets_test: Dict mapping dataset names to Data objects (or ``None``
                placeholders when ``eval_only`` and PCA/smoothing grid defers loading).
            seed: Random seed.
            
        Returns:
            Results dict with metrics.
        """
        set_seed(seed)
        self._current_seed = int(seed)
        
        if self.args.init_checkpoint and self.args.resume_checkpoint:
            raise ValueError("--init_checkpoint and --resume_checkpoint are mutually exclusive.")
        
        if self.is_main:
            mode = "eval" if self.args.eval_only else "train"
            run_name = get_wandb_run_name(self.base_name, seed, mode)
            wandb_kwargs = {
                "project": self.args.wandb_project,
                "name": run_name,
                "config": vars(self.args),
                "reinit": True,
            }
            if not self.args.eval_only and self.args.resume_checkpoint:
                # Reuse the saved wandb run only when resuming training weights.
                try:
                    existing_run_id = load_wandb_run_id(self.checkpoint_dir)
                except (FileNotFoundError, OSError):
                    existing_run_id = ""
                if existing_run_id:
                    wandb_kwargs["id"] = existing_run_id
                    wandb_kwargs["resume"] = "allow"
            wandb.init(**wandb_kwargs)
            if not self.args.eval_only:
                save_training_args(self.checkpoint_dir, self.args)
                save_wandb_run_id(self.checkpoint_dir, wandb.run.id)
                write_training_status(
                    self.checkpoint_dir,
                    status="running",
                    last_completed_epoch=-1,
                    max_epochs=self.args.max_epochs,
                )
                if getattr(self.args, "skip_eval", False):
                    define_async_wandb_metrics()
        
        model_cfg = build_wander_config(self.args)
        model = Wander(model_cfg).to(self.device)
        
        if self.world_size > 1 and not self.args.eval_only:
            model = DDP(model, device_ids=[self.rank], find_unused_parameters=True)
        
        if self.args.init_checkpoint:
            raw_model = model.module if isinstance(model, DDP) else model
            state_dict = torch.load(self.args.init_checkpoint, map_location=self.device)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            if self.args.reset_rw_parameters:
                model_sd = raw_model.state_dict()
                filtered = merge_init_checkpoint_state_dict(model_sd, state_dict)
                raw_model.load_state_dict(filtered, strict=False)
                if self.is_main:
                    n_rw = len(rw_parameter_keys(model_sd))
                    print(
                        f"Initialized from {self.args.init_checkpoint} (--reset_rw_parameters): "
                        f"{n_rw} RW-related tensors at fresh init, {len(filtered)} tensors loaded from checkpoint."
                    )
            else:
                raw_model.load_state_dict(
                    filter_obsolete_checkpoint_keys(state_dict), strict=False
                )
                if self.is_main:
                    print(f"Initialized model from checkpoint: {self.args.init_checkpoint}")
        
        if self.args.eval_only:
            if not self.args.init_checkpoint:
                raise ValueError("--eval_only requires --init_checkpoint to specify which model to evaluate.")
            
            eval_model = model.module if isinstance(model, DDP) else model
            test_metrics: Dict[str, Dict[str, float]] = {}

            for ds_name, ds_data in datasets_test.items():
                try:
                    ds_eval = ds_data

                    loader = MultiGraphLoader(
                        bundles={ds_name: ds_eval},
                        split="test",
                        **self._eval_multigraph_loader_kwargs(
                            seed,
                            max_eval_samples=self.args.max_eval_samples_eval_only,
                        ),
                    )
                    ds_metrics = self.evaluate(loader, eval_model, "test")

                    test_metrics[ds_name] = ds_metrics[ds_name]

                    test_metrics["aggregate"] = aggregate_metrics(
                        {k: v for k, v in test_metrics.items() if k != "aggregate"}
                    )

                    if self.is_main:
                        log_dict = {}
                        for k, v in test_metrics[ds_name].items():
                            if k.startswith("coverage/"):
                                cov_key = k[len("coverage/"):]
                                log_dict[f"coverage/test/{ds_name}/{cov_key}"] = v
                            elif should_log_metric_to_wandb(k):
                                log_dict[f"test/{ds_name}/{k}"] = v
                        wandb.log(log_dict)
                        summary = ", ".join(
                            f"{k}={v:.4f}"
                            for k, v in test_metrics[ds_name].items()
                            if not k.startswith("coverage/")
                            and should_log_metric_to_wandb(k)
                        )
                        print(f"[eval_only] {ds_name} done: {summary}")

                except Exception:
                    if self.is_main:
                        print(f"[eval_only] {ds_name} FAILED:\n{traceback.format_exc()}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if dist.is_initialized() and self.world_size > 1:
                        dist.abort(1)

            if self.is_main:
                if test_metrics:
                    metrics_path = self._save_eval_metrics(
                        seed=seed,
                        test_metrics=test_metrics,
                    )
                    print(f"Saved eval metrics to {metrics_path}")
                wandb.finish()
                return {"test": test_metrics} if test_metrics else {"test": {}}
            return {}
        
        raw_model = model.module if isinstance(model, DDP) else model

        start_epoch = 0
        opt_state = None
        if self.args.resume_checkpoint:
            ckpt = torch.load(self.args.resume_checkpoint, map_location=self.device)
            raw_model.load_state_dict(
                filter_obsolete_checkpoint_keys(ckpt["model_state_dict"]), strict=False
            )
            opt_state = ckpt.get("optimizer_state_dict")
            start_epoch = ckpt["epoch"] + 1
            if self.is_main:
                print(
                    f"Resumed from {self.args.resume_checkpoint}, "
                    f"starting at epoch {start_epoch}"
                )

        trainable_params = [p for p in raw_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        if opt_state is not None:
            n_ckpt_opt = sum(len(g["params"]) for g in opt_state["param_groups"])
            n_model = len(trainable_params)
            if n_ckpt_opt == n_model:
                optimizer.load_state_dict(opt_state)
            elif self.is_main:
                print(
                    f"Warning: checkpoint optimizer tracks {n_ckpt_opt} parameters "
                    f"but the current trainable set has {n_model}; using a fresh optimizer "
                    f"(model weights and epoch counter are still restored)."
                )
        
        results = self.trainer(datasets_train, datasets_test, model, optimizer, seed, start_epoch=start_epoch)
        
        if self.is_main:
            if not getattr(self.args, "skip_eval", False) and results.get("best_metrics") is not None:
                best_log = {"best/epoch": results["best_metrics"]["epoch"]}
                for prefix, source_metrics in [
                    ("val", results["best_metrics"].get("val")),
                    ("test", results["best_metrics"].get("test")),
                ]:
                    if source_metrics is None:
                        continue
                    for name, metrics in source_metrics.items():
                        for k, v in metrics.items():
                            if k.startswith("coverage/"):
                                cov_key = k[len("coverage/"):]
                                best_log[f"coverage/best/{prefix}/{name}/{cov_key}"] = v
                            elif should_log_metric_to_wandb(k):
                                best_log[f"best/{prefix}/{name}/{k}"] = v
                wandb.log(best_log)
            wandb.finish()
        
        return results
    
    def run_many_seeds(
        self
    ) -> Dict[str, any]:
        """Run experiment over multiple seeds and aggregate results.
            
        Returns:
            Aggregated results across all seeds.
        """
        all_results = []
        
        for seed in self._eval_seeds():
            if self.is_main:
                print(f"\n{'='*50}")
                print(f"Running seed {seed}")
                print(f"{'='*50}")

            datasets_train, datasets_test = self.load_datasets(seed)

            if self.is_main:
                print(f"{'-'*50}")
                print(f"RUNNING EXPERIMENT")
            results = self.run_single_seed(datasets_train, datasets_test, seed)
            all_results.append(results)
        
        if self.is_main:
            if self.args.eval_only:
                aggregated = aggregate_seed_results(
                    [r["test"] for r in all_results],
                )
            elif getattr(self.args, "skip_eval", False):
                print("\nTraining finished with --skip_eval.")
                return {}
            else:
                aggregated = {
                    "val": aggregate_seed_results(
                        [r["best_metrics"]["val"] for r in all_results],
                    ),
                }
                test_splits = [r["best_metrics"].get("test") for r in all_results]
                if all(split is not None for split in test_splits):
                    aggregated["test"] = aggregate_seed_results(test_splits)
            
            print(f"\n{'='*50}")
            print("Final Results (averaged over seeds)")
            print(f"{'='*50}")
            
            if self.args.eval_only:
                print_aggregated_results(aggregated, "test")
                seeds = self._eval_seeds()
                agg_path = os.path.join(self.checkpoint_dir, "metrics_aggregate.json")
                save_json(
                    agg_path,
                    {
                        "experiment_name": self.base_name,
                        "seeds": seeds,
                        "init_checkpoint": getattr(self.args, "init_checkpoint", None),
                        "nc_split_index": getattr(self.args, "nc_split_index", None),
                        "metrics": aggregated,
                    },
                )
                print(f"Saved seed-aggregated metrics to {agg_path}")
                if len(seeds) > 1:
                    flat: Dict[str, Dict[str, float]] = {}
                    for graph_name, metrics in aggregated.items():
                        entry: Dict[str, float] = {}
                        for metric_name, stats in metrics.items():
                            entry[metric_name] = float(stats["mean"])
                            entry[f"{metric_name}_std"] = float(stats["std"])
                        entry["n_seeds"] = float(len(seeds))
                        flat[graph_name] = entry
                    save_json(
                        os.path.join(self.checkpoint_dir, "metrics.json"),
                        {
                            "experiment_name": self.base_name,
                            "seeds": seeds,
                            "init_checkpoint": getattr(
                                self.args, "init_checkpoint", None
                            ),
                            "nc_split_index": getattr(
                                self.args, "nc_split_index", None
                            ),
                            "metrics": flat,
                        },
                    )
                    txt_path = os.path.join(self.checkpoint_dir, "metrics.txt")
                    lines = []
                    for ds_name in sorted(flat):
                        if ds_name == "aggregate":
                            continue
                        summary = self._format_eval_metric_summary(flat[ds_name])
                        lines.append(f"[eval_only] {ds_name} done: {summary}")
                    if "aggregate" in flat:
                        summary = self._format_eval_metric_summary(flat["aggregate"])
                        lines.append(f"[eval_only] aggregate: {summary}")
                    with open(txt_path, "w", encoding="utf-8") as handle:
                        handle.write("\n".join(lines))
                        handle.write("\n")
                    print(
                        f"Wrote seed-averaged metrics.json over {len(seeds)} seeds"
                    )
            else:
                print_aggregated_results(aggregated["val"], "val")
                if "test" in aggregated:
                    print_aggregated_results(aggregated["test"], "test")
            
            return aggregated
        
        return None
    
    


