"""On-the-fly synthetic graph loader with background prefetch."""

from __future__ import annotations

import copy
import os
import pickle
import queue
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data

from data.dataset import effective_pca_target_dim
from data.prior.dataset import SCMPrior
from data.prior_to_pyg import prior_to_pyg_data, prior_to_pyg_lp_data
from experiment.utils import capture_rng_state, restore_rng_state, seed_rngs


_SENTINEL = object()
_CUDA_UNSET = object()


def close_synthetic_prefetch(loader: object, *, kill_workers: bool = True) -> None:
    """Close a ``SyntheticGraphLoader`` or a wrapper that owns one."""
    seen: list = []
    for obj in (loader, getattr(loader, "synthetic", None)):
        if obj is None or obj in seen:
            continue
        seen.append(obj)
        closer = getattr(obj, "close", None)
        if callable(closer):
            closer(kill_workers=kill_workers)

StepItem = Tuple[str, Tensor, Data]
LoaderYield = Union[StepItem, List[StepItem]]
_SYNTHETIC_TASKS: Tuple[str, ...] = ("node_cls", "lp")

# Process-pool worker state (spawn children only).
_MP_LOADER: Optional["SyntheticGraphLoader"] = None


def _mp_ping() -> int:
    """Force spawn workers to start while CUDA_VISIBLE_DEVICES is still cleared."""
    return os.getpid()


def _mp_init_loader(snapshot: dict) -> None:
    """Build a CPU-only loader in a spawn worker. Must stay at module scope for pickle."""
    global _MP_LOADER
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("GRAPH_WALKER_NUM_THREADS", "1")
    torch.set_num_threads(1)
    seed = (os.getpid() ^ time.time_ns()) & 0x7FFFFFFF
    np.random.seed(seed)
    torch.manual_seed(seed)
    prior = SCMPrior(
        classification_only=snapshot["classification_only"],
        fixed_hp=snapshot["fixed_hp"],
        sampled_hp=snapshot["sampled_hp"],
        device="cpu",
    )
    kw = dict(snapshot["loader"])
    kw["prefetch_workers"] = 1
    kw["prefetch_mp"] = False
    _MP_LOADER = SyntheticGraphLoader(prior=prior, **kw)


def _mp_generate_one(task: str) -> StepItem:
    assert _MP_LOADER is not None
    return _MP_LOADER._generate_one(task)


class SyntheticGraphLoader:
    """Generate fresh synthetic graphs with pipelined prefetch.

    A background thread continuously generates graphs via ``SCMPrior`` so
    that the next batch is ready by the time the model finishes the current
    training/evaluation step.

    Parameters
    ----------
    prior : SCMPrior
        Configured prior dataset generator (``batch_size=1``).
    batch_size : int
        Number of node indices per batch (sampled from the test split).
    batch_per_epoch : int
        Number of training steps per epoch (= number of optimizer steps).
    pca_target_dim_node_syn : int
        Passed through to ``prior_to_pyg_data`` for node-classification graphs.
    pca_target_dim_link_syn : int
        Passed through to ``prior_to_pyg_lp_data`` for synthetic LP graphs.
    is_training : bool
        If False (e.g. synthetic validation), node batches are always sampled uniformly.
    balanced_training : bool
        If True and ``is_training`` is True, use class-balanced test-node batches; otherwise uniform.
    prefetch : int
        Queue capacity for in-flight / ready graphs (backpressure).
    prefetch_workers : int
        Number of background generator workers. ``1`` is the original
        single-producer path. Use more when sampling is slower than a step.
    prefetch_mp : bool
        If True and ``prefetch_workers > 1``, workers are spawn processes
        (needed when sampling is GIL-bound, e.g. GraphPFN). Otherwise threads.
    num_graphs_per_step : int
        Global number of synthetic graphs per optimizer step. When ``ddp_world_size > 1``
        and ``is_training``, each rank yields ``num_graphs_per_step // ddp_world_size``
        graphs per step (must divide evenly). When ``ddp_world_size == 1``, one yield
        contains ``num_graphs_per_step`` graphs (list if ``> 1``, else singleton).
    ddp_world_size : int
        Number of DDP ranks; ``1`` disables per-rank splitting (default).
    """

    def __init__(
        self,
        prior: SCMPrior,
        batch_size: int,
        batch_per_epoch: int,
        pca_target_dim_node_syn: int = 32,
        pca_target_dim_link_syn: int = 32,
        adaptive_pca_node_threshold: int = 0,
        is_training: bool = True,
        balanced_training: bool = True,
        prefetch: int = 2,
        prefetch_workers: int = 1,
        prefetch_mp: bool = True,
        num_graphs_per_step: int = 1,
        ddp_world_size: int = 1,
        row_wise_norming: bool = False,
        row_norm_mode: str = "l2",
        dummy_features: bool = False,
        ignore_features: bool = False,
        link_pred_prob: float = 0.0,
        batch_size_link: int = 16,
        pca_before_normalization: bool = True,
        graphland_different_transform: bool = False,
        graphland_categorical_as_ordinals: bool = False,
        drop_constant_train_features: bool = False,
        final_inductive_zscore: bool = False,
        drop_feature_indices=None,
        deterministic_seed: Optional[int] = None,
    ):
        # drop_feature_indices applies only to real graphs; accepted for kwargs parity.
        del drop_feature_indices
        self.prior = prior
        self.batch_size = batch_size
        self.batch_size_link = max(1, int(batch_size_link))
        self.link_pred_prob = float(link_pred_prob)
        self.batch_per_epoch = batch_per_epoch
        self.pca_target_dim_node_syn = pca_target_dim_node_syn
        self.pca_target_dim_link_syn = pca_target_dim_link_syn
        self.adaptive_pca_node_threshold = adaptive_pca_node_threshold
        self.dummy_features = dummy_features
        self.ignore_features = ignore_features
        self.is_training = is_training
        self.balanced_training = balanced_training
        self.prefetch = max(1, prefetch)
        self.prefetch_workers = max(1, int(prefetch_workers))
        self.prefetch_mp = bool(prefetch_mp)
        self.num_graphs_per_step = max(1, num_graphs_per_step)
        self.row_wise_norming = row_wise_norming
        self.row_norm_mode = row_norm_mode
        self.pca_before_normalization = pca_before_normalization
        self.graphland_different_transform = graphland_different_transform
        self.graphland_categorical_as_ordinals = graphland_categorical_as_ordinals
        self.drop_constant_train_features = drop_constant_train_features
        self.final_inductive_zscore = final_inductive_zscore
        self._epoch = 0
        self._mp_executor: Optional[ProcessPoolExecutor] = None
        self._saved_cuda_visible = _CUDA_UNSET
        # Fixed-task-set mode (validation). See _iter_deterministic.
        self.deterministic_seed = deterministic_seed
        self._det_items: Optional[List[StepItem]] = None

        ws = max(1, ddp_world_size)
        if is_training and ws > 1:
            if self.num_graphs_per_step % ws != 0:
                raise ValueError(
                    f"num_graphs_per_step ({self.num_graphs_per_step}) must be divisible "
                    f"by ddp_world_size ({ws}) for multi-GPU synthetic training."
                )
            self._graphs_per_rank_step = self.num_graphs_per_step // ws
        else:
            self._graphs_per_rank_step = self.num_graphs_per_step

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _hide_cuda_for_spawn(self) -> None:
        """Clear CUDA_VISIBLE_DEVICES before spawn so workers never init a GPU.

        Parent CUDA is already initialized; clearing the env var does not
        unbind it. Keep it cleared while the pool is alive so respawned
        workers also stay CPU-only.
        """
        if self._saved_cuda_visible is _CUDA_UNSET:
            self._saved_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    def _restore_cuda_visible(self) -> None:
        saved = self._saved_cuda_visible
        if saved is _CUDA_UNSET:
            return
        if saved is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(saved)
        self._saved_cuda_visible = _CUDA_UNSET

    def close(self, kill_workers: bool = True) -> None:
        """Drop or shut down spawn prefetch workers.

        ``kill_workers=False`` leaves the pool warm for the next epoch (DGL
        import + first-sample cost is large). Use True before process exit or
        when replacing the loader.
        """
        if not kill_workers:
            return
        ex = self._mp_executor
        self._mp_executor = None
        if ex is not None:
            ex.shutdown(wait=True, cancel_futures=True)
            if self.is_training:
                print(
                    f"[SyntheticGraphLoader] closed spawn prefetch workers (epoch {self._epoch})",
                    flush=True,
                )
        self._restore_cuda_visible()

    def __len__(self) -> int:
        return self.batch_per_epoch

    def _preprocess_kwargs(self) -> dict:
        return {
            "row_wise_norming": self.row_wise_norming,
            "row_norm_mode": self.row_norm_mode,
            "dummy_features": self.dummy_features,
            "ignore_features": self.ignore_features,
            "pca_before_normalization": self.pca_before_normalization,
            "graphland_different_transform": self.graphland_different_transform,
            "graphland_categorical_as_ordinals": self.graphland_categorical_as_ordinals,
            "drop_constant_train_features": self.drop_constant_train_features,
            "final_inductive_zscore": self.final_inductive_zscore,
        }

    def _generate_one(self, task: str = "node_cls") -> StepItem:
        preprocess = self._preprocess_kwargs()
        if task == "lp":
            X, lp_meta = self.prior.get_batch(task_type="lp")
            pca_dim = effective_pca_target_dim(
                self.pca_target_dim_link_syn,
                int(lp_meta["num_nodes"]),
                self.adaptive_pca_node_threshold,
            )
            data = prior_to_pyg_lp_data(
                X, lp_meta,
                pca_target_dim=pca_dim,
                **preprocess,
            )
            return ("synthetic", self._sample_lp_batch(data), data)

        graph, X, y, d, _seq_len, train_size = self.prior.get_batch()
        pca_dim = effective_pca_target_dim(
            self.pca_target_dim_node_syn,
            graph.num_nodes,
            self.adaptive_pca_node_threshold,
        )
        data = prior_to_pyg_data(
            graph, X, y, d.item(), train_size,
            pca_target_dim=pca_dim,
            **preprocess,
        )
        if not self.is_training:
            batch = self._sample_random_batch(data)
        elif self.balanced_training:
            batch = self._sample_balanced_batch(data)
        else:
            batch = self._sample_random_batch(data)
        return ("synthetic", batch, data)

    def _task_coins(self) -> List[str]:
        """Per-step task sequence, seeded by epoch (shared across DDP ranks).

        Draws ``batch_per_epoch`` tasks from {node_cls, lp} with probability
        ``link_pred_prob`` (remainder is node classification). Same mix is used
        for synthetic validation.
        """
        p_lp = max(0.0, float(self.link_pred_prob))
        if p_lp <= 0.0:
            return ["node_cls"] * self.batch_per_epoch
        g = torch.Generator()
        g.manual_seed(self._epoch)
        u = torch.rand(self.batch_per_epoch, generator=g)
        return ["lp" if ui < p_lp else "node_cls" for ui in u.tolist()]

    def _task_sequence(self) -> List[str]:
        coins = self._task_coins()
        tasks: List[str] = []
        for task in coins:
            tasks.extend([task] * self._graphs_per_rank_step)
        return tasks

    def _log_queue_wait(self, wait_s: float) -> None:
        if wait_s > 1e-3:
            print(
                f"[SyntheticGraphLoader] queue empty (epoch {self._epoch}); "
                f"waited {wait_s:.4f}s",
                flush=True,
            )

    def _yield_epoch(self, next_item) -> Iterator[LoaderYield]:
        for _ in range(self.batch_per_epoch):
            if self._graphs_per_rank_step == 1:
                item = next_item()
                if item is _SENTINEL:
                    break
                yield item
            else:
                group: List[StepItem] = []
                short = False
                for _g in range(self._graphs_per_rank_step):
                    item = next_item()
                    if item is _SENTINEL:
                        short = True
                        break
                    group.append(item)
                if short:
                    if group:
                        yield group
                    break
                yield group

    def _yield_epoch_by_task(self, next_for_task) -> Iterator[LoaderYield]:
        """Consume any ready graph of the step's task type (DDP-aligned, not HOL-ordered)."""
        coins = self._task_coins()
        for task in coins:
            if self._graphs_per_rank_step == 1:
                item = next_for_task(task)
                if item is _SENTINEL:
                    break
                yield item
            else:
                group: List[StepItem] = []
                short = False
                for _g in range(self._graphs_per_rank_step):
                    item = next_for_task(task)
                    if item is _SENTINEL:
                        short = True
                        break
                    group.append(item)
                if short:
                    if group:
                        yield group
                    break
                yield group

    def _mp_snapshot(self) -> dict:
        return {
            "classification_only": bool(self.prior.classification_only),
            "fixed_hp": copy.deepcopy(self.prior.fixed_hp),
            "sampled_hp": copy.deepcopy(self.prior.sampled_hp),
            "loader": {
                "batch_size": self.batch_size,
                "batch_per_epoch": self.batch_per_epoch,
                "pca_target_dim_node_syn": self.pca_target_dim_node_syn,
                "pca_target_dim_link_syn": self.pca_target_dim_link_syn,
                "adaptive_pca_node_threshold": self.adaptive_pca_node_threshold,
                "is_training": self.is_training,
                "balanced_training": self.balanced_training,
                "prefetch": 2,
                "num_graphs_per_step": 1,
                "ddp_world_size": 1,
                "row_wise_norming": self.row_wise_norming,
                "row_norm_mode": self.row_norm_mode,
                "dummy_features": self.dummy_features,
                "ignore_features": self.ignore_features,
                "link_pred_prob": self.link_pred_prob,
                "batch_size_link": self.batch_size_link,
                "pca_before_normalization": self.pca_before_normalization,
                "graphland_different_transform": self.graphland_different_transform,
                "graphland_categorical_as_ordinals": self.graphland_categorical_as_ordinals,
                "drop_constant_train_features": self.drop_constant_train_features,
                "final_inductive_zscore": self.final_inductive_zscore,
                "deterministic_seed": self.deterministic_seed,
            },
        }

    def _can_use_process_pool(self) -> bool:
        if not self.prefetch_mp or self.prefetch_workers <= 1:
            return False
        try:
            pickle.dumps(self._mp_snapshot())
        except Exception:
            return False
        return True

    def _iter_deterministic(self) -> Iterator[LoaderYield]:
        """Yield the *same* task set on every pass, generated once and cached.

        The default paths generate in a background producer thread that draws
        from the process-global numpy/torch RNGs -- the very RNGs the model
        consumes for walk sampling in the main thread. Seeding once before the
        loop therefore does not pin the task set: the two streams interleave in
        a timing-dependent order, so each epoch scores a different draw and the
        resampling noise swamps the real epoch-to-epoch change.

        Generating up front inside a single save/seed/restore region gives the
        task set its own private stream, independent of whatever the model does
        between items. Caching it then keeps the set identical across epochs
        (and skips regeneration entirely on later epochs).
        """
        if self._det_items is None:
            outer = capture_rng_state()
            try:
                seed_rngs(int(self.deterministic_seed))
                tasks = self._task_sequence()
                self._det_items = [self._generate_one(task) for task in tasks]
            finally:
                restore_rng_state(outer)

        items = iter(self._det_items)

        def _next_item() -> StepItem:
            return next(items, _SENTINEL)

        yield from self._yield_epoch(_next_item)

    def __iter__(self) -> Iterator[LoaderYield]:
        if self.deterministic_seed is not None:
            yield from self._iter_deterministic()
            return

        if self.prefetch_workers > 1 and self.is_training:
            use_mp = self._can_use_process_pool()
            if self.is_training:
                mode = "spawn-mp" if use_mp else "threads"
                print(
                    f"[SyntheticGraphLoader] prefetch={self.prefetch} "
                    f"workers={self.prefetch_workers} mode={mode} "
                    f"(epoch {self._epoch})",
                    flush=True,
                )
            if use_mp:
                yield from self._iter_process_pool()
            else:
                yield from self._iter_thread_pool()
            return

        q: queue.Queue = queue.Queue(maxsize=self.prefetch)
        error: list = []
        tasks = self._task_sequence()

        def _producer():
            try:
                for task in tasks:
                    q.put(self._generate_one(task))
            except Exception as exc:
                error.append(exc)
            finally:
                q.put(_SENTINEL)

        thread = threading.Thread(target=_producer, daemon=True)
        thread.start()

        def _next_item() -> StepItem:
            t0 = time.perf_counter()
            it = q.get()
            self._log_queue_wait(time.perf_counter() - t0)
            if error:
                raise error[0]
            return it

        try:
            yield from self._yield_epoch(_next_item)
        finally:
            while True:
                item = q.get()
                if item is _SENTINEL:
                    break
            thread.join()

        if error:
            raise error[0]

    def _iter_process_pool(self) -> Iterator[LoaderYield]:
        """Spawn processes; consume any ready graph of the needed task type."""
        tasks = self._task_sequence()
        n_tasks = len(tasks)
        snapshot = self._mp_snapshot()
        if self._mp_executor is None:
            self._hide_cuda_for_spawn()
            ctx = torch.multiprocessing.get_context("spawn")
            self._mp_executor = ProcessPoolExecutor(
                max_workers=self.prefetch_workers,
                mp_context=ctx,
                initializer=_mp_init_loader,
                initargs=(snapshot,),
            )
            ping = [
                self._mp_executor.submit(_mp_ping)
                for _ in range(self.prefetch_workers)
            ]
            wait(ping)
            for fut in ping:
                fut.result()
        executor = self._mp_executor
        inflight: dict = {}
        ready: dict[str, deque] = {t: deque() for t in _SYNTHETIC_TASKS}
        next_submit = 0

        def _ready_for(task: str) -> deque:
            return ready[task]

        def _inflight_for(task: str) -> int:
            return sum(1 for t in inflight.values() if t == task)

        def _buffered() -> int:
            return len(inflight) + sum(len(q) for q in ready.values())

        def _pump(need: Optional[str] = None) -> None:
            nonlocal next_submit
            while next_submit < n_tasks:
                starved = (
                    need is not None
                    and len(_ready_for(need)) == 0
                    and _inflight_for(need) == 0
                    and _buffered() >= self.prefetch
                )
                if _buffered() >= self.prefetch and not starved:
                    return
                if starved:
                    match = next(
                        (i for i in range(next_submit, n_tasks) if tasks[i] == need),
                        None,
                    )
                    if match is None:
                        return
                    if match != next_submit:
                        tasks[next_submit], tasks[match] = (
                            tasks[match],
                            tasks[next_submit],
                        )
                task = tasks[next_submit]
                inflight[executor.submit(_mp_generate_one, task)] = task
                next_submit += 1
                if starved:
                    return

        def _collect(timeout: Optional[float] = None) -> None:
            if not inflight:
                return
            done, _ = wait(
                tuple(inflight.keys()),
                timeout=timeout,
                return_when=FIRST_COMPLETED,
            )
            for fut in done:
                task = inflight.pop(fut)
                _ready_for(task).append(fut.result())

        def _next_for_task(task: str):
            ready = _ready_for(task)
            t0 = time.perf_counter()
            _pump(need=task)
            while not ready:
                if not inflight:
                    _pump(need=task)
                    if not inflight:
                        return _SENTINEL
                _collect(timeout=0.5)
                _pump(need=task)
            item = ready.popleft()
            _pump(need=task)
            self._log_queue_wait(time.perf_counter() - t0)
            return item

        try:
            yield from self._yield_epoch_by_task(_next_for_task)
        finally:
            for fut in list(inflight):
                fut.cancel()
                inflight.pop(fut, None)
            # Keep the spawn pool warm; train_epoch / complexity rebuild
            # call close(kill_workers=True) when the loader is discarded.

    def _iter_thread_pool(self) -> Iterator[LoaderYield]:
        """Thread fallback: type-bucketed queues so a slow graph does not HOL-block."""
        tasks = self._task_sequence()
        n_tasks = len(tasks)
        claimed = [False] * n_tasks
        queues: dict[str, queue.Queue] = {t: queue.Queue() for t in _SYNTHETIC_TASKS}
        job_cv = threading.Condition()
        error: list = []
        stop = threading.Event()
        inflight_counts: dict[str, int] = {t: 0 for t in _SYNTHETIC_TASKS}

        def _q_for(task: str) -> queue.Queue:
            return queues[task]

        def _claim() -> Optional[str]:
            with job_cv:
                while True:
                    if stop.is_set() or error:
                        return None
                    found = None
                    for i, task in enumerate(tasks):
                        if claimed[i]:
                            continue
                        ready = _q_for(task).qsize()
                        inflight = inflight_counts[task]
                        if ready + inflight < self.prefetch:
                            claimed[i] = True
                            found = task
                            inflight_counts[task] += 1
                            break
                    if found is not None:
                        return found
                    if all(claimed):
                        return None
                    job_cv.wait(timeout=0.2)

        def _worker() -> None:
            while True:
                task = _claim()
                if task is None:
                    return
                try:
                    item = self._generate_one(task)
                except Exception as exc:
                    error.append(exc)
                    with job_cv:
                        job_cv.notify_all()
                    return
                _q_for(task).put(item)
                with job_cv:
                    inflight_counts[task] -= 1
                    job_cv.notify_all()

        threads = [
            threading.Thread(target=_worker, daemon=True)
            for _ in range(self.prefetch_workers)
        ]
        for thread in threads:
            thread.start()

        def _next_for_task(task: str):
            q = _q_for(task)
            t0 = time.perf_counter()
            while q.empty() and not error:
                with job_cv:
                    job_cv.wait(timeout=0.2)
            if error and q.empty():
                raise error[0]
            item = q.get()
            with job_cv:
                job_cv.notify_all()
            self._log_queue_wait(time.perf_counter() - t0)
            return item

        try:
            yield from self._yield_epoch_by_task(_next_for_task)
        finally:
            stop.set()
            with job_cv:
                job_cv.notify_all()
            for thread in threads:
                thread.join()

    # ------------------------------------------------------------------
    # Sampling (test split — matches inference on held-out nodes)
    # ------------------------------------------------------------------

    def _sample_lp_batch(self, data: Data) -> Tensor:
        """Sample a 2D link-prediction query batch from held-out (test) edges.

        Returns ``[n, 3]`` with ``n = batch_size_link``.
        """
        test_e = data.lp_test_edges
        E = int(test_e.shape[1])
        if E == 0:
            return torch.zeros(0, 3, dtype=torch.long)

        n = min(self.batch_size_link, E)
        idx = torch.randperm(E)[:n]
        h = test_e[0, idx].clone()
        t = test_e[1, idx].clone()
        if bool(getattr(data, "undirected_lp", False)):
            flip = torch.rand(n) < 0.5
            h, t = torch.where(flip, t, h), torch.where(flip, h, t)
        rel = torch.zeros(n, dtype=torch.long)
        return torch.stack([h, rel, t], dim=1)

    def _sample_random_batch(
        self, data: Data, batch_size: Optional[int] = None,
    ) -> Tensor:
        """Sample test nodes uniformly at random (order shuffled)."""
        test_indices = data.test_mask.nonzero(as_tuple=True)[0]
        n = min(self.batch_size if batch_size is None else int(batch_size),
                len(test_indices))
        idx = torch.randperm(len(test_indices))[:n]
        return test_indices[idx]

    def _sample_balanced_batch(
        self, data: Data, batch_size: Optional[int] = None,
    ) -> Tensor:
        """Sample a class-balanced set of test-node indices."""
        test_indices = data.test_mask.nonzero(as_tuple=True)[0]
        labels = data.y[test_indices].argmax(dim=1)
        num_classes = data.y.shape[1]
        batch_size = min(self.batch_size if batch_size is None else int(batch_size),
                         len(test_indices))

        class_to_nodes: dict[int, Tensor] = {}
        for c in range(num_classes):
            mask = labels == c
            class_to_nodes[c] = test_indices[mask]

        non_empty = [c for c in range(num_classes) if len(class_to_nodes[c]) > 0]
        k = len(non_empty)

        if batch_size <= k:
            chosen = [non_empty[i] for i in torch.randperm(k)[:batch_size].tolist()]
            selected = []
            for c in chosen:
                pool = class_to_nodes[c]
                selected.append(pool[torch.randint(0, len(pool), (1,))])
        else:
            per_class = batch_size // k
            remainder = batch_size - per_class * k
            selected = []
            for c in non_empty:
                pool = class_to_nodes[c]
                selected.append(pool[torch.randint(0, len(pool), (per_class,))])
            if remainder > 0:
                extra = [non_empty[i] for i in torch.randperm(k)[:remainder].tolist()]
                for c in extra:
                    pool = class_to_nodes[c]
                    selected.append(pool[torch.randint(0, len(pool), (1,))])

        out = torch.cat(selected, dim=0)
        return out[torch.randperm(out.shape[0])]
