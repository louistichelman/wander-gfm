"""CPU walk prefetch and async H2D transfer for multi-sample eval ensemble."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar

import torch

WalkTuple = tuple
T = TypeVar("T")
_SENTINEL = object()


@dataclass
class EnsemblePrefetchStats:
    """Per-forward breakdown for pipelined ensemble eval."""

    n_samples: int = 0
    wait_get_ms: float = 0.0
    transfer_ms: float = 0.0
    forward_ms: float = 0.0
    produce_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.wait_get_ms + self.transfer_ms + self.forward_ms

    @property
    def wait_get_pct(self) -> float:
        return 100.0 * self.wait_get_ms / self.total_ms if self.total_ms else 0.0

    @property
    def transfer_pct(self) -> float:
        return 100.0 * self.transfer_ms / self.total_ms if self.total_ms else 0.0

    @property
    def forward_pct(self) -> float:
        return 100.0 * self.forward_ms / self.total_ms if self.total_ms else 0.0

    def merge_producer(self, produce_ms: float, n_produced: int) -> None:
        self.produce_ms = produce_ms
        self.n_samples = n_produced

    def summary_text(self) -> str:
        lines = [
            "Ensemble prefetch profile (per forward, ms)",
            "=" * 72,
            f"n_samples: {self.n_samples}",
            f"wait_get_ms: {self.wait_get_ms:.2f} ({self.wait_get_pct:.1f}% of active time)",
            f"transfer_ms: {self.transfer_ms:.2f} ({self.transfer_pct:.1f}% of active time)",
            f"forward_ms: {self.forward_ms:.2f} ({self.forward_pct:.1f}% of active time)",
            f"produce_ms (background): {self.produce_ms:.2f}",
            f"active_total_ms: {self.total_ms:.2f}",
        ]
        if self.n_samples > 0:
            lines.append(
                f"per_sample: wait={self.wait_get_ms / self.n_samples:.2f} "
                f"transfer={self.transfer_ms / self.n_samples:.2f} "
                f"forward={self.forward_ms / self.n_samples:.2f} "
                f"produce={self.produce_ms / self.n_samples:.2f}"
            )
        if self.wait_get_ms > 0 and self.produce_ms > 0:
            overlap_ratio = 1.0 - (self.wait_get_ms / self.produce_ms)
            lines.append(
                f"estimated_walk_overlap: {max(0.0, overlap_ratio) * 100:.1f}% "
                f"(1 - wait_get/produce; 100% = prefetch always ahead)"
            )
        return "\n".join(lines) + "\n"


class WalkPrefetcher:
    """Background producer for CPU walk tuples consumed by the main eval thread."""

    def __init__(
        self,
        generate_fn: Callable[[], WalkTuple],
        n_samples: int,
        queue_depth: int = 2,
        stats: Optional[EnsemblePrefetchStats] = None,
    ):
        self._generate_fn = generate_fn
        self._n_samples = n_samples
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, queue_depth))
        self._errors: list[BaseException] = []
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._closed = False
        self._stats = stats
        self._produce_ms_total = 0.0
        self._n_produced = 0

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(target=self._producer, daemon=True)
        self._thread.start()

    def _producer(self) -> None:
        try:
            for _ in range(self._n_samples):
                t0 = time.perf_counter()
                self._queue.put(self._generate_fn())
                self._produce_ms_total += (time.perf_counter() - t0) * 1000.0
                self._n_produced += 1
        except BaseException as exc:
            self._errors.append(exc)
        finally:
            self._queue.put(_SENTINEL)

    def get(self) -> WalkTuple:
        if self._errors:
            raise self._errors[0]
        if not self._started:
            self.start()
        item = self._queue.get()
        if item is _SENTINEL:
            if self._errors:
                raise self._errors[0]
            raise RuntimeError("WalkPrefetcher exhausted before all samples were consumed.")
        return item

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._thread is not None:
            self._thread.join(timeout=120.0)
        if self._stats is not None:
            self._stats.merge_producer(self._produce_ms_total, self._n_produced)
        if self._errors:
            raise self._errors[0]


class AsyncWalkTransfer:
    """Pinned host-to-device walk transfer on a side CUDA stream."""

    def __init__(self, device: torch.device):
        self.device = device
        self._stream: Optional[torch.cuda.Stream] = None
        self._gpu_tuple: Optional[WalkTuple] = None
        if device.type == "cuda":
            self._stream = torch.cuda.Stream(device=device)

    @staticmethod
    def _to_device(cpu_tuple: WalkTuple, device: torch.device, *, non_blocking: bool) -> WalkTuple:
        if device.type != "cuda":
            return cpu_tuple
        out = []
        for t in cpu_tuple:
            if not t.is_pinned():
                t = t.pin_memory()
            out.append(t.to(device, non_blocking=non_blocking))
        return tuple(out)

    def transfer_sync(self, cpu_tuple: WalkTuple) -> WalkTuple:
        return self._to_device(cpu_tuple, self.device, non_blocking=False)

    def submit_async(self, cpu_tuple: WalkTuple) -> None:
        if self._stream is None:
            self._gpu_tuple = cpu_tuple
            return
        with torch.cuda.stream(self._stream):
            self._gpu_tuple = self._to_device(cpu_tuple, self.device, non_blocking=True)

    def wait(self) -> WalkTuple:
        if self._stream is not None:
            self._stream.synchronize()
        if self._gpu_tuple is None:
            raise RuntimeError("AsyncWalkTransfer.wait() called with no pending transfer.")
        out = self._gpu_tuple
        self._gpu_tuple = None
        return out


def run_pipelined_ensemble(
    prefetcher: WalkPrefetcher,
    device: torch.device,
    forward_fn: Callable[[WalkTuple], T],
    n_samples: int,
) -> tuple[list[T], EnsemblePrefetchStats]:
    """Run pipelined ensemble: overlap CPU walk prefetch/H2D with GPU forward."""
    stats = prefetcher._stats or EnsemblePrefetchStats()
    transfer = AsyncWalkTransfer(device)
    outputs: list[T] = []

    prefetcher.start()
    try:
        t0 = time.perf_counter()
        cpu_cur = prefetcher.get()
        stats.wait_get_ms += (time.perf_counter() - t0) * 1000.0

        for i in range(n_samples):
            if i == 0:
                t0 = time.perf_counter()
                gpu_cur = transfer.transfer_sync(cpu_cur)
                stats.transfer_ms += (time.perf_counter() - t0) * 1000.0
            else:
                t0 = time.perf_counter()
                gpu_cur = transfer.wait()
                stats.transfer_ms += (time.perf_counter() - t0) * 1000.0

            if i + 1 < n_samples:
                t0 = time.perf_counter()
                cpu_next = prefetcher.get()
                stats.wait_get_ms += (time.perf_counter() - t0) * 1000.0
                transfer.submit_async(cpu_next)

            t0 = time.perf_counter()
            outputs.append(forward_fn(gpu_cur))
            stats.forward_ms += (time.perf_counter() - t0) * 1000.0

        stats.n_samples = n_samples
        return outputs, stats
    finally:
        prefetcher.close()
