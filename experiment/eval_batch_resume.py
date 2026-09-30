"""Resume for eval-only NC / LP batch loops.

Zero-shot eval of large graphs (e.g. CITY_PARIS) can run for longer.
This module writes a small per-rank checkpoint of already scored batches
under ``<run>/eval_batch_resume/`` so a restarted job can skip those
forwards and still produce the same aggregate metrics.

The train-KV cache is not persisted (it can be tens of GB); only scored
batch outputs are. Random-walk ensemble draws on remaining batches are
independent of the saved ones, so mixing two job attempts is statistically
valid even though it is not bit-identical to a single uninterrupted run.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, List, Optional

import torch
from torch import Tensor

EVAL_BATCH_RESUME_DIRNAME = "eval_batch_resume"
EVAL_BATCH_RESUME_VERSION = 1


def eval_batch_resume_dir(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, EVAL_BATCH_RESUME_DIRNAME)


def eval_batch_resume_path(
    checkpoint_dir: str,
    *,
    dataset_name: str,
    split: str,
    seed: int,
    rank: int,
) -> str:
    safe_ds = str(dataset_name).replace(os.sep, "_")
    return os.path.join(
        eval_batch_resume_dir(checkpoint_dir),
        f"{split}_{safe_ds}_seed{int(seed)}_rank{int(rank)}.pt",
    )


def build_eval_batch_fingerprint(
    args: Any,
    *,
    dataset_name: str,
    split: str,
    seed: int,
    rank: int,
    world_size: int,
    n_eval_items: int,
) -> Dict[str, Any]:
    """Args that change batch order or scores. Mismatch discards the file."""
    return {
        "version": EVAL_BATCH_RESUME_VERSION,
        "dataset_name": str(dataset_name),
        "split": str(split),
        "seed": int(seed),
        "rank": int(rank),
        "world_size": int(world_size),
        "n_eval_items": int(n_eval_items),
        "init_checkpoint": getattr(args, "init_checkpoint", None),
        "nc_split_index": getattr(args, "nc_split_index", None),
        "eval_batch_size_node": getattr(args, "eval_batch_size_node", None),
        "nc_proximity_batching": bool(getattr(args, "nc_proximity_batching", True)),
        "nc_proximity_batch_max_radius": getattr(
            args, "nc_proximity_batch_max_radius", None
        ),
        "nc_proximity_batch_seed": getattr(args, "nc_proximity_batch_seed", None),
        "wander_walk_num": getattr(args, "wander_walk_num", None),
        "wander_eval_walk_num": getattr(args, "wander_eval_walk_num", None),
        "wander_keep_train_free_p": getattr(args, "wander_keep_train_free_p", None),
        "wander_walk_len": getattr(args, "wander_walk_len", None),
        "wander_max_walk_len": getattr(args, "wander_max_walk_len", None),
        "wander_test_samples": getattr(args, "wander_test_samples", None),
        "wander_train_kv_passes": getattr(args, "wander_train_kv_passes", None),
        "wander_train_kv_chunk_size": getattr(args, "wander_train_kv_chunk_size", None),
        "wander_adaptive_walks": bool(getattr(args, "wander_adaptive_walks", False)),
        "pca_target_dim": getattr(args, "pca_target_dim", None),
        "drop_constant_train_features": bool(
            getattr(args, "drop_constant_train_features", False)
        ),
        "final_inductive_zscore": bool(getattr(args, "final_inductive_zscore", False)),
    }


def atomic_torch_save(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _cpu_tensor(value: Optional[Tensor]) -> Optional[Tensor]:
    if value is None:
        return None
    return value.detach().cpu()


def cat_chunks(chunks: List[Tensor]) -> Optional[Tensor]:
    if not chunks:
        return None
    return torch.cat(chunks, dim=0)


def load_eval_batch_resume(
    path: str,
    fingerprint: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if int(payload.get("version", -1)) != EVAL_BATCH_RESUME_VERSION:
        return None
    stored_fp = payload.get("fingerprint")
    if stored_fp != fingerprint:
        return None
    next_idx = int(payload.get("next_batch_idx", 0) or 0)
    if next_idx <= 0:
        return None
    return payload


def save_eval_batch_resume(
    path: str,
    *,
    fingerprint: Dict[str, Any],
    next_batch_idx: int,
    items_done: int,
    correct_count: int,
    total_count: int,
    batch_sizes: List[int],
    node_ids: Optional[Tensor] = None,
    cls_targets: Optional[Tensor] = None,
    cls_probas: Optional[Tensor] = None,
    logits_sum: Optional[Tensor] = None,
    logits_count: Optional[Tensor] = None,
    coverage_sums: Optional[Dict[str, float]] = None,
    coverage_count: int = 0,
) -> None:
    payload: Dict[str, Any] = {
        "version": EVAL_BATCH_RESUME_VERSION,
        "fingerprint": fingerprint,
        "next_batch_idx": int(next_batch_idx),
        "items_done": int(items_done),
        "correct_count": int(correct_count),
        "total_count": int(total_count),
        "batch_sizes": [int(x) for x in batch_sizes],
        "coverage_sums": {str(k): float(v) for k, v in (coverage_sums or {}).items()},
        "coverage_count": int(coverage_count),
    }
    payload["node_ids"] = _cpu_tensor(node_ids)
    payload["cls_targets"] = _cpu_tensor(cls_targets)
    payload["cls_probas"] = _cpu_tensor(cls_probas)
    payload["logits_sum"] = _cpu_tensor(logits_sum)
    payload["logits_count"] = _cpu_tensor(logits_count)
    atomic_torch_save(path, payload)


def clear_eval_batch_resume(path: str) -> None:
    for candidate in (path, f"{path}.tmp"):
        try:
            os.remove(candidate)
        except FileNotFoundError:
            pass


def truncate_resume_payload(
    payload: Dict[str, Any],
    n_batches: int,
) -> Optional[Dict[str, Any]]:
    """Keep only the first ``n_batches`` scored batches.

    Accumulated logits cannot be truncated without per-batch node ids, so a
    shorter prefix returns ``None`` (caller starts fresh).
    """
    n_batches = int(n_batches)
    if n_batches <= 0:
        return None
    sizes = [int(x) for x in payload.get("batch_sizes") or []]
    if n_batches > len(sizes):
        return None
    if n_batches == int(payload.get("next_batch_idx", 0) or 0):
        return payload
    if payload.get("logits_sum") is not None or payload.get("logits_count") is not None:
        return None
    keep_items = sum(sizes[:n_batches])
    out = dict(payload)
    out["next_batch_idx"] = n_batches
    out["batch_sizes"] = sizes[:n_batches]
    out["items_done"] = keep_items
    if out.get("total_count") is not None:
        out["total_count"] = keep_items
    for key in ("node_ids", "cls_targets", "cls_probas"):
        tensor = out.get(key)
        if tensor is not None:
            out[key] = tensor[:keep_items]
    targets = out.get("cls_targets")
    probas = out.get("cls_probas")
    if (
        targets is not None
        and probas is not None
        and probas.shape[0] == targets.shape[0]
    ):
        out["correct_count"] = int((probas.argmax(dim=1) == targets).sum().item())
    return out


def verify_resume_node_ids(
    batches: Iterable[Tensor],
    stored_node_ids: Optional[Tensor],
    n_batches: int,
) -> bool:
    """True if the first ``n_batches`` loader batches match ``stored_node_ids``."""
    if stored_node_ids is None:
        return True
    seen: List[Tensor] = []
    for i, batch in enumerate(batches):
        if i >= n_batches:
            break
        if batch.numel() > 0:
            seen.append(batch.detach().cpu().view(-1))
    if not seen:
        return int(stored_node_ids.numel()) == 0
    actual = torch.cat(seen, dim=0)
    stored = stored_node_ids.detach().cpu().view(-1)
    return actual.shape == stored.shape and bool(torch.equal(actual, stored))


def eval_batch_resume_enabled_from_env(args: Any) -> bool:
    if not getattr(args, "eval_only", False):
        return False
    if not getattr(args, "eval_batch_resume", True):
        return False
    force = os.environ.get("FORCE_FRESH", "0").strip().lower()
    if force in {"1", "true", "yes"}:
        return False
    return True


def resolve_and_validate_resume(
    *,
    path: str,
    fingerprint: Dict[str, Any],
    query_loader: Any,
    world_size: int,
    device: torch.device,
) -> tuple[int, Optional[Dict[str, Any]]]:
    """Load a compatible resume file and agree ``resume_from`` across DDP ranks.

    Ranks without a matching file report ``0``. The agreed start is the MIN
    ``next_batch_idx``. A node-id mismatch against the current loader starts
    fresh.
    """
    loaded = load_eval_batch_resume(path, fingerprint)
    local_next = int(loaded["next_batch_idx"]) if loaded else 0
    resume_from = local_next

    if world_size > 1:
        import torch.distributed as dist

        nxt = torch.tensor([local_next], dtype=torch.long, device=device)
        dist.all_reduce(nxt, op=dist.ReduceOp.MIN)
        resume_from = int(nxt.item())
        if loaded is not None and resume_from < local_next:
            loaded = truncate_resume_payload(loaded, resume_from)
            if loaded is None:
                resume_from = 0
        if loaded is None:
            resume_from = 0
        agreed = torch.tensor([resume_from], dtype=torch.long, device=device)
        dist.all_reduce(agreed, op=dist.ReduceOp.MIN)
        resume_from = int(agreed.item())
        if resume_from <= 0:
            loaded = None

    if loaded is None or resume_from <= 0:
        return 0, None

    stored_ids = loaded.get("node_ids")
    loader_batches = [batch for batch, _data in query_loader]
    ok = verify_resume_node_ids(loader_batches, stored_ids, resume_from)
    if world_size > 1:
        import torch.distributed as dist

        flag = torch.tensor([1 if ok else 0], dtype=torch.long, device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        ok = bool(flag.item())
    if not ok:
        return 0, None
    return resume_from, loaded


def apply_resume_payload(
    loaded: Dict[str, Any],
) -> Dict[str, Any]:
    """Unpack a validated resume payload into evaluate() accumulators."""
    cls_targets = loaded.get("cls_targets")
    cls_probas = loaded.get("cls_probas")
    out: Dict[str, Any] = {
        "items_done": int(loaded.get("items_done", 0) or 0),
        "correct_count": int(loaded.get("correct_count", 0) or 0),
        "total_count": int(loaded.get("total_count", 0) or 0),
        "batch_sizes": [int(x) for x in (loaded.get("batch_sizes") or [])],
        "cls_targets_chunks": [cls_targets] if cls_targets is not None else [],
        "cls_proba_chunks": [cls_probas] if cls_probas is not None else [],
        "node_id_chunks": [],
        "coverage_sums": {
            str(k): float(v) for k, v in (loaded.get("coverage_sums") or {}).items()
        },
        "coverage_count": int(loaded.get("coverage_count", 0) or 0),
    }
    if loaded.get("node_ids") is not None:
        out["node_id_chunks"] = [loaded["node_ids"]]
    return out


@contextmanager
def eval_preempt_flag() -> Iterator[Dict[str, bool]]:
    """Set ``flag['stop']`` on SIGTERM / SIGINT so the caller can flush and exit."""
    import signal

    flag = {"stop": False}

    def _handler(signum: int, _frame: Any) -> None:
        flag["stop"] = True

    old_term = signal.getsignal(signal.SIGTERM)
    old_int = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)
    try:
        yield flag
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
