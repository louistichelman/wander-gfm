"""Poll training checkpoint directories and run async per-epoch evaluation.

Supports a single run (--checkpoint_dir) or round-robin across several runs
(--checkpoint_dirs): one pending epoch per turn, rebuild Wander and reattach
W&B (same run id via resume=must) when switching.
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import wandb

from config.constants import WANDB_API_KEY
from models import Wander
from config.arguments import build_wander_config, fill_missing_arg_defaults, _expand_dataset_list
from data.lp_rank_stats import is_per_relation_metric, should_log_metric_to_wandb

from .experiment import Experiment
from .utils import (
    EVAL_PROGRESS_FILENAME,
    TRAINING_ARGS_FILENAME,
    TRAINING_STATUS_FILENAME,
    define_async_wandb_metrics,
    get_wandb_run_name,
    load_json,
    load_wandb_run_id,
    namespace_from_dict,
    save_epoch_eval_metrics,
    save_final_test_metrics,
    save_json,
    write_prior_complexity_override,
)


EPOCH_CHECKPOINT_RE = re.compile(r"model_seed(\d+)_epoch(\d+)\.pt$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch checkpoint directory(ies) and evaluate new epoch checkpoints.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Single experiment checkpoint directory (contains training_args.json).",
    )
    parser.add_argument(
        "--checkpoint_dirs",
        type=str,
        nargs="+",
        default=None,
        help="Multiple checkpoint directories to watch in round-robin (one epoch per turn).",
    )
    parser.add_argument("--poll_interval", type=int, default=300, help="Seconds between idle polls.")
    parser.add_argument("--seed", type=int, default=0, help="Seed subdirectory to evaluate.")
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=0,
        help="First epoch checkpoint to evaluate (skips earlier epochs).",
    )
    parser.add_argument(
        "--max_epoch",
        type=int,
        default=None,
        help="Stop after this epoch is evaluated. Defaults to training metadata max_epochs.",
    )
    parser.add_argument(
        "--latest_only",
        action="store_true",
        help=(
            "Only evaluate the newest available checkpoint each turn for every run, "
            "skipping intermediate epochs that were never evaluated."
        ),
    )
    parser.add_argument(
        "--latest_only_dirs",
        type=str,
        nargs="+",
        default=None,
        help="Subset of checkpoint dirs that use --latest_only (per-run). Absolute or basename.",
    )
    parser.add_argument(
        "--test_datasets",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Override test_datasets from training_args.json (e.g. when a resumed "
            "training job overwrote it with the default test dataset)."
        ),
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="val",
        choices=("val", "test"),
        help="Split used for per-epoch eval (default: val). Overridable per run via --eval_split_dirs.",
    )
    parser.add_argument(
        "--eval_split_dirs",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Per-run eval split overrides as RUN_NAME:val|test "
            "(e.g. pt3:test). Absolute dirs also accepted."
        ),
    )
    args = parser.parse_args()
    if not args.checkpoint_dir and not args.checkpoint_dirs:
        parser.error("one of --checkpoint_dir or --checkpoint_dirs is required")
    if args.checkpoint_dir and args.checkpoint_dirs:
        parser.error("use either --checkpoint_dir or --checkpoint_dirs, not both")
    return args


def _resolve_checkpoint_dirs(args: argparse.Namespace) -> List[str]:
    if args.checkpoint_dirs:
        return [os.path.abspath(d) for d in args.checkpoint_dirs]
    return [os.path.abspath(args.checkpoint_dir)]


def _dir_matches(checkpoint_dir: str, spec: str) -> bool:
    """Match an absolute path or a run-name basename against a CLI spec."""
    spec_abs = os.path.abspath(spec)
    if checkpoint_dir == spec_abs or checkpoint_dir.rstrip("/") == spec_abs.rstrip("/"):
        return True
    return os.path.basename(os.path.normpath(checkpoint_dir)) == os.path.basename(
        os.path.normpath(spec)
    )


def _latest_only_for_dir(checkpoint_dir: str, args: argparse.Namespace) -> bool:
    if args.latest_only:
        return True
    for spec in args.latest_only_dirs or []:
        if _dir_matches(checkpoint_dir, spec):
            return True
    return False


def _eval_split_for_dir(checkpoint_dir: str, args: argparse.Namespace) -> str:
    """Resolve per-run eval split; defaults to --eval_split."""
    for spec in args.eval_split_dirs or []:
        if ":" not in spec:
            raise ValueError(
                f"--eval_split_dirs entries must be RUN:val|test, got {spec!r}"
            )
        run_spec, split = spec.rsplit(":", 1)
        split = split.strip().lower()
        if split not in ("val", "test"):
            raise ValueError(
                f"--eval_split_dirs split must be val or test, got {split!r} in {spec!r}"
            )
        if _dir_matches(checkpoint_dir, run_spec.strip()):
            return split
    return str(getattr(args, "eval_split", "val") or "val")


@dataclass
class WatchTarget:
    checkpoint_dir: str
    latest_only: bool
    eval_split: str = "val"
    training_args: Optional[argparse.Namespace] = None
    effective_start: int = 0
    max_epoch: int = 0
    progress: Dict[str, Any] = field(default_factory=dict)
    finished: bool = False
    prepared: bool = False

    @property
    def run_name(self) -> str:
        return os.path.basename(os.path.normpath(self.checkpoint_dir))


def _val_score_from_wandb_row(
    row: Dict[str, Any],
    checkpoint_metric: str = "auto",
) -> Optional[float]:
    """Reconstruct the validation score stored in a wandb history row."""

    def _get(metric: str) -> Optional[float]:
        key = f"val/aggregate/{metric}"
        value = row.get(key)
        if value is None:
            return None
        fv = float(value)
        if fv != fv:
            return None
        return fv

    metric = str(checkpoint_metric or "auto")
    if metric != "auto":
        chosen = _get(metric)
        if chosen is not None:
            return chosen
        for fallback in ("accuracy", "mrr", "roc_auc"):
            if fallback == metric:
                continue
            fb = _get(fallback)
            if fb is not None:
                return fb
    else:
        parts: list[float] = []
        for m in ("mrr", "accuracy"):
            fv = _get(m)
            if fv is not None:
                parts.append(fv)
        if parts:
            return sum(parts) / len(parts)
    for key, value in row.items():
        if (
            isinstance(key, str)
            and key.startswith("val/aggregate/")
            and not is_per_relation_metric(key[len("val/aggregate/") :])
            and value is not None
        ):
            fv = float(value)
            if fv == fv:
                return fv
    return None


def _resume_progress_from_wandb(project: str, run_id: str) -> Dict[str, Any]:
    """Read watcher progress from an existing wandb run history."""
    api = wandb.Api()
    run = None
    runs = list(api.runs(project, filters={"id": run_id}))
    if runs:
        run = runs[0]
    else:
        try:
            run = api.run(run_id)
        except Exception:
            run = None
    if run is None:
        print(
            f"Could not find wandb run id={run_id} in project={project}; "
            "using local eval_progress.json only."
        )
        return {}
        watcher_prefixes = ("val/", "val_synthetic/", "train_eval/")
    last_evaluated_epoch = -1
    best_val_score: Optional[float] = None
    best_epoch: Optional[int] = None

    for row in run.scan_history():
        step = row.get("_step")
        if step is None:
            continue
        epoch = int(step)
        has_watcher_metrics = any(
            isinstance(key, str)
            and any(key.startswith(prefix) for prefix in watcher_prefixes)
            and row.get(key) is not None
            for key in row.keys()
        )
        if not has_watcher_metrics:
            continue

        last_evaluated_epoch = max(last_evaluated_epoch, epoch)
        ckpt_metric = "auto"
        val_score = _val_score_from_wandb_row(row, checkpoint_metric=ckpt_metric)
        if val_score is not None and (best_val_score is None or val_score > best_val_score):
            best_val_score = val_score
            best_epoch = epoch

    summary = dict(run.summary)
    final_test_done = "best/epoch" in summary or any(
        str(key).startswith("best/test/") for key in summary
    )

    progress: Dict[str, Any] = {"resume_source": "wandb"}
    if last_evaluated_epoch >= 0:
        progress["last_evaluated_epoch"] = last_evaluated_epoch
    if best_epoch is not None:
        progress["best_epoch"] = best_epoch
    if best_val_score is not None:
        progress["best_val_score"] = best_val_score
    if final_test_done:
        progress["final_test_done"] = True
    return progress


def _merge_eval_progress(local: Dict[str, Any], remote: Dict[str, Any]) -> Dict[str, Any]:
    """Merge local checkpoint progress with progress reconstructed from wandb."""
    merged = dict(local)
    sources = []
    if local:
        sources.append("eval_progress.json")
    if remote:
        sources.append("wandb")

    local_last = int(local.get("last_evaluated_epoch", -1))
    remote_last = int(remote.get("last_evaluated_epoch", -1))
    merged["last_evaluated_epoch"] = max(local_last, remote_last)

    local_best = local.get("best_val_score")
    remote_best = remote.get("best_val_score")
    if local_best is not None and remote_best is not None:
        if remote_best > local_best:
            merged["best_val_score"] = remote_best
            merged["best_epoch"] = remote.get("best_epoch")
        else:
            merged["best_val_score"] = local_best
            merged["best_epoch"] = local.get("best_epoch")
    elif remote_best is not None:
        merged["best_val_score"] = remote_best
        merged["best_epoch"] = remote.get("best_epoch")
    elif local_best is not None:
        merged["best_val_score"] = local_best
        merged["best_epoch"] = local.get("best_epoch")

    merged["final_test_done"] = bool(
        local.get("final_test_done") or remote.get("final_test_done")
    )
    merged["resume_sources"] = sources
    return merged


def _ensure_best_checkpoint(
    checkpoint_dir: str,
    seed: int,
    progress: Dict[str, Any],
    checkpoints: Dict[int, str],
) -> None:
    """Recreate model_seed*_best.pt from the best epoch checkpoint if missing."""
    best_path = _best_checkpoint_path(checkpoint_dir, seed)
    if os.path.isfile(best_path):
        return

    best_epoch = progress.get("best_epoch")
    if best_epoch is None:
        return

    ckpt_path = checkpoints.get(int(best_epoch))
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        print(
            f"Best epoch {best_epoch} is recorded but checkpoint is missing; "
            f"cannot recreate {best_path}"
        )
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    torch.save(state_dict, best_path)
    print(f"Recreated best checkpoint for epoch {best_epoch} at {best_path}")


def _load_training_args(checkpoint_dir: str) -> argparse.Namespace:
    path = os.path.join(checkpoint_dir, TRAINING_ARGS_FILENAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing {TRAINING_ARGS_FILENAME} in {checkpoint_dir}. "
            "Start training first so metadata is written."
        )
    # Backfill CLI defaults for flags added since this run's training_args.json
    # was written (e.g. resuming/watching an older checkpoint dir).
    return fill_missing_arg_defaults(namespace_from_dict(load_json(path)))


def _training_args_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, TRAINING_ARGS_FILENAME)


def _load_training_status(checkpoint_dir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(checkpoint_dir, TRAINING_STATUS_FILENAME)
    if not os.path.isfile(path):
        return None
    return load_json(path)


def _eval_progress_filename(eval_split: str = "val") -> str:
    """Val keeps the legacy filename; other splits get a dedicated progress file."""
    if eval_split == "val":
        return EVAL_PROGRESS_FILENAME
    return f"eval_progress_{eval_split}.json"


def _load_eval_progress(
    checkpoint_dir: str, eval_split: str = "val"
) -> Dict[str, Any]:
    path = os.path.join(checkpoint_dir, _eval_progress_filename(eval_split))
    if not os.path.isfile(path):
        return {}
    return load_json(path)


def _save_eval_progress(
    checkpoint_dir: str, progress: Dict[str, Any], eval_split: str = "val"
) -> None:
    save_json(
        os.path.join(checkpoint_dir, _eval_progress_filename(eval_split)), progress
    )


def _resolve_max_epoch(
    checkpoint_dir: str,
    args: argparse.Namespace,
    cli_max_epoch: Optional[int],
) -> int:
    if cli_max_epoch is not None:
        return cli_max_epoch
    status = _load_training_status(checkpoint_dir)
    if status is not None and "max_epochs" in status:
        return int(status["max_epochs"]) - 1
    return int(args.max_epochs) - 1


def _discover_epoch_checkpoints(checkpoint_dir: str, seed: int) -> Dict[int, str]:
    checkpoints: Dict[int, str] = {}
    if not os.path.isdir(checkpoint_dir):
        return checkpoints
    for name in os.listdir(checkpoint_dir):
        match = EPOCH_CHECKPOINT_RE.match(name)
        if not match:
            continue
        if int(match.group(1)) != seed:
            continue
        epoch = int(match.group(2))
        checkpoints[epoch] = os.path.join(checkpoint_dir, name)
    return checkpoints


def _checkpoint_is_stable(
    path: str,
    seen: Dict[str, tuple[float, int]],
    *,
    min_age_seconds: float = 60.0,
) -> bool:
    """Return True when a checkpoint file is safe to load.

    Files older than ``min_age_seconds`` are treated as stable immediately
    (training finished writing them). Fresh files require two identical polls.
    """
    try:
        stat = os.stat(path)
    except FileNotFoundError:
        return False
    signature = (stat.st_mtime, stat.st_size)
    if (time.time() - stat.st_mtime) >= min_age_seconds:
        seen[path] = signature
        return True
    previous = seen.get(path)
    seen[path] = signature
    return previous == signature


def _log_eval_to_wandb(log_dict: Dict[str, Any], epoch: int, chunk_size: int = 25) -> None:
    """Log async watcher eval metrics on the shared training run.

    Uses ``eval_epoch`` (via define_async_wandb_metrics) as the x-axis for val/*
    charts. Does not pass ``step=epoch`` because the training job may already have
    advanced the run's global step past this epoch (e.g. inline val),
    which would cause W&B to drop the watcher's full-benchmark metrics.
    """
    if not log_dict:
        return

    payload = dict(log_dict)
    payload.pop("epoch", None)
    payload["eval_epoch"] = epoch
    # Last-line defense: never push num_queries or per-relation LP breakdowns.
    drop_keys = []
    for key in payload:
        if key in ("eval_epoch", "prior_complexity", "prior_complexity_next",
                   "prior_complexity_apply_from_epoch") or key.startswith("coverage/"):
            continue
        # Keys look like ``test/DS/mrr/12`` or ``val/DS/num_queries``.
        leaf = key.split("/", 2)[-1] if key.count("/") >= 2 else key.rsplit("/", 1)[-1]
        if not should_log_metric_to_wandb(leaf):
            drop_keys.append(key)
    for key in drop_keys:
        payload.pop(key, None)

    items = list(payload.items())
    for offset in range(0, len(items), chunk_size):
        chunk = dict(items[offset : offset + chunk_size])
        is_last = offset + chunk_size >= len(items)
        wandb.log(chunk, commit=is_last)


def _init_wandb(checkpoint_dir: str, args: argparse.Namespace, seed: int) -> None:
    run_id = load_wandb_run_id(checkpoint_dir)
    base_name = os.path.basename(os.path.normpath(checkpoint_dir))
    wandb.init(
        project=args.wandb_project,
        id=run_id,
        resume="must",
        name=get_wandb_run_name(base_name, seed, "train"),
        config=vars(args),
        reinit=True,
    )
    define_async_wandb_metrics()


def _finish_wandb_quietly() -> None:
    try:
        if wandb.run is not None:
            wandb.finish()
    except Exception as exc:
        print(f"WARNING: wandb.finish failed: {exc}")


def _maybe_write_prior_complexity_override(
    checkpoint_dir: str,
    *,
    args: argparse.Namespace,
    epoch: int,
    prior_complexity: float,
    syn_acc: Optional[float],
) -> None:
    if not getattr(args, "prior_complexity_adapt", False):
        return
    if syn_acc is None or syn_acc <= 0.65 or prior_complexity >= 1.0:
        return
    new_complexity = min(1.0, prior_complexity + 0.05)
    write_prior_complexity_override(
        checkpoint_dir,
        apply_from_epoch=epoch + 1,
        prior_complexity=new_complexity,
        based_on_epoch=epoch,
        syn_acc=syn_acc,
    )
    print(
        f"Prior complexity will increase to {new_complexity:.2f} from epoch {epoch + 1} "
        f"(syn_acc={syn_acc:.4f} at epoch {epoch})"
    )
    wandb.log({
        "eval_epoch": epoch,
        "prior_complexity_next": new_complexity,
        "prior_complexity_apply_from_epoch": epoch + 1,
        "val_synthetic/aggregate/accuracy": syn_acc,
    })


def _best_checkpoint_path(checkpoint_dir: str, seed: int) -> str:
    return os.path.join(checkpoint_dir, f"model_seed{seed}_best.pt")


def _run_final_test_if_ready(
    experiment: Experiment,
    model: torch.nn.Module,
    datasets_test: Dict[str, Any],
    seed: int,
    progress: Dict[str, Any],
    checkpoint_dir: str,
) -> bool:
    if progress.get("final_test_done"):
        return True

    best_state_path = _best_checkpoint_path(checkpoint_dir, seed)
    if not os.path.isfile(best_state_path):
        print(f"Waiting for best checkpoint at {best_state_path}")
        return False

    best_state = torch.load(best_state_path, map_location=experiment.device)
    test_metrics = experiment.run_final_test_eval(
        model,
        datasets_test,
        seed,
        best_state,
    )

    best_log: Dict[str, Any] = {"best/epoch": progress.get("best_epoch")}
    val_metrics = progress.get("best_val_metrics")
    if val_metrics is not None:
        for name, metrics in val_metrics.items():
            for k, v in metrics.items():
                if k.startswith("coverage/"):
                    cov_key = k[len("coverage/") :]
                    best_log[f"coverage/best/val/{name}/{cov_key}"] = v
                elif should_log_metric_to_wandb(k):
                    best_log[f"best/val/{name}/{k}"] = v
    for name, metrics in test_metrics.items():
        for k, v in metrics.items():
            if k.startswith("coverage/"):
                cov_key = k[len("coverage/") :]
                best_log[f"coverage/best/test/{name}/{cov_key}"] = v
            elif should_log_metric_to_wandb(k):
                best_log[f"best/test/{name}/{k}"] = v
    wandb.log(best_log)

    metrics_path = save_final_test_metrics(
        checkpoint_dir,
        best_epoch=int(progress.get("best_epoch", -1)),
        test_metrics=test_metrics,
    )
    print(f"Saved final test metrics to {metrics_path}")

    progress["final_test_done"] = True
    progress["best_test_metrics"] = test_metrics
    _save_eval_progress(checkpoint_dir, progress, eval_split="val")
    print(f"Final test eval complete for best epoch {progress.get('best_epoch')}")
    return True


def _normalize_training_args(
    checkpoint_dir: str,
    training_args: argparse.Namespace,
    test_datasets_override: Optional[List[str]],
) -> argparse.Namespace:
    training_args.run_name = os.path.basename(os.path.normpath(checkpoint_dir))
    training_args.save_load_path = os.path.dirname(checkpoint_dir)
    training_args.skip_eval = True
    if test_datasets_override is not None:
        training_args.test_datasets = _expand_dataset_list(list(test_datasets_override))
        print(
            f"[{training_args.run_name}] Using test_datasets override "
            f"({len(training_args.test_datasets)} datasets)"
        )
    return training_args


def _try_prepare_target(
    target: WatchTarget,
    *,
    start_epoch: int,
    max_epoch: Optional[int],
    test_datasets_override: Optional[List[str]],
    seed: int,
) -> bool:
    """Load training_args / progress for a target. Returns False if not ready yet."""
    if target.prepared:
        return True
    if not os.path.isfile(_training_args_path(target.checkpoint_dir)):
        return False

    training_args = _load_training_args(target.checkpoint_dir)
    training_args = _normalize_training_args(
        target.checkpoint_dir, training_args, test_datasets_override
    )
    resolved_max = _resolve_max_epoch(target.checkpoint_dir, training_args, max_epoch)

    # Val watcher may resume from wandb; other splits use a dedicated progress
    # file and must not inherit val's last_evaluated_epoch from the shared run.
    wandb_progress: Dict[str, Any] = {}
    if target.eval_split == "val":
        try:
            run_id = load_wandb_run_id(target.checkpoint_dir)
            wandb_progress = _resume_progress_from_wandb(
                training_args.wandb_project, run_id
            )
        except FileNotFoundError:
            print(
                f"[{target.run_name}] No wandb_run_id.txt yet; "
                "resume uses eval_progress.json only."
            )
        except Exception as exc:
            print(
                f"[{target.run_name}] Could not resume from wandb ({exc}); "
                "using local progress only."
            )

    local_progress = _load_eval_progress(
        target.checkpoint_dir, eval_split=target.eval_split
    )
    progress = _merge_eval_progress(local_progress, wandb_progress)
    effective_start = max(
        start_epoch,
        int(progress.get("last_evaluated_epoch", -1)) + 1,
    )
    progress["start_epoch"] = effective_start
    progress["eval_split"] = target.eval_split
    progress.setdefault("max_epoch", resolved_max)
    progress.setdefault("final_test_done", False)
    _save_eval_progress(
        target.checkpoint_dir, progress, eval_split=target.eval_split
    )

    checkpoints = _discover_epoch_checkpoints(target.checkpoint_dir, seed)
    if target.eval_split == "val":
        _ensure_best_checkpoint(target.checkpoint_dir, seed, progress, checkpoints)

    resume_sources = progress.get("resume_sources", [])
    print(
        f"[{target.run_name}] Resume: last_evaluated_epoch="
        f"{progress.get('last_evaluated_epoch', -1)}, "
        f"best_epoch={progress.get('best_epoch')}, "
        f"final_test_done={progress.get('final_test_done', False)}, "
        f"sources={resume_sources or ['none']}, "
        f"latest_only={target.latest_only}, "
        f"eval_split={target.eval_split}"
    )

    target.training_args = training_args
    target.effective_start = effective_start
    target.max_epoch = resolved_max
    target.progress = progress
    target.finished = bool(progress.get("final_test_done"))
    target.prepared = True
    return True


def _dataset_names_from_args(training_args: argparse.Namespace) -> Tuple[List[str], List[str]]:
    train_names = list(getattr(training_args, "train_datasets", None) or [])
    test_names = list(getattr(training_args, "test_datasets", None) or [])
    return train_names, test_names


class DatasetCache:
    """Shared GraphBundle cache keyed by dataset name (union across watched runs)."""

    def __init__(self) -> None:
        self.bundles: Dict[str, Any] = {}
        self._loader: Optional[Experiment] = None

    def _ensure_loader(self, training_args: argparse.Namespace) -> Experiment:
        if self._loader is None:
            # Loader only needs data_dir / preprocess knobs; dataset lists set per load.
            loader_args = argparse.Namespace(**vars(training_args))
            loader_args.train_datasets = []
            loader_args.test_datasets = []
            self._loader = Experiment(loader_args, rank=0, world_size=1)
        return self._loader

    def ensure_names(
        self,
        names: Set[str],
        training_args: argparse.Namespace,
        seed: int,
    ) -> None:
        missing = sorted(n for n in names if n not in self.bundles)
        if not missing:
            return
        experiment = self._ensure_loader(training_args)
        print(f"Dataset cache: loading {len(missing)} new dataset(s): {missing}")
        experiment.args.train_datasets = []
        experiment.args.test_datasets = missing
        _, loaded_test = experiment.load_datasets(seed)
        self.bundles.update(loaded_test)

    def subset(self, names: List[str]) -> Dict[str, Any]:
        missing = [n for n in names if n not in self.bundles]
        if missing:
            raise KeyError(f"Datasets not in cache: {missing}")
        return {n: self.bundles[n] for n in names}


def _pending_epochs(
    target: WatchTarget,
    seed: int,
    seen_stable: Dict[str, tuple[float, int]],
) -> Tuple[List[int], Dict[int, str]]:
    """Return stable pending epochs (possibly truncated by latest_only) and ckpt map."""
    progress = _load_eval_progress(
        target.checkpoint_dir, eval_split=target.eval_split
    )
    target.progress = progress
    last_evaluated = int(progress.get("last_evaluated_epoch", -1))
    checkpoints = _discover_epoch_checkpoints(target.checkpoint_dir, seed)
    pending = [
        epoch
        for epoch in sorted(checkpoints)
        if target.effective_start <= epoch <= target.max_epoch and epoch > last_evaluated
    ]
    if target.latest_only and pending:
        skipped = pending[:-1]
        if skipped:
            print(
                f"[{target.run_name}] Skipping unevaluated epochs "
                f"{skipped[0]}..{skipped[-1]} (--latest_only); "
                f"evaluating epoch {pending[-1]} only"
            )
        pending = [pending[-1]]

    stable: List[int] = []
    for epoch in pending:
        ckpt_path = checkpoints[epoch]
        if not _checkpoint_is_stable(ckpt_path, seen_stable):
            print(f"[{target.run_name}] Epoch {epoch} checkpoint not stable yet: {ckpt_path}")
            break
        stable.append(epoch)
    return stable, checkpoints


def _ready_for_final(target: WatchTarget) -> bool:
    progress = _load_eval_progress(
        target.checkpoint_dir, eval_split=target.eval_split
    )
    last_evaluated = int(progress.get("last_evaluated_epoch", -1))
    training_status = _load_training_status(target.checkpoint_dir)
    training_complete = (
        training_status is not None and training_status.get("status") == "complete"
    )
    target_epoch = min(
        target.max_epoch,
        int(training_status.get("last_completed_epoch", target.max_epoch))
        if training_status is not None
        else target.max_epoch,
    )
    return last_evaluated >= target.max_epoch or (
        training_complete and last_evaluated >= target_epoch
    )


def _release_model(model: Optional[torch.nn.Module]) -> None:
    if model is None:
        return
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_model(
    training_args: argparse.Namespace,
) -> Tuple[Experiment, torch.nn.Module]:
    experiment = Experiment(training_args, rank=0, world_size=1)
    model_cfg = build_wander_config(training_args)
    model = Wander(model_cfg).to(experiment.device)
    return experiment, model


def _evaluate_one_epoch(
    target: WatchTarget,
    *,
    epoch: int,
    ckpt_path: str,
    experiment: Experiment,
    model: torch.nn.Module,
    datasets_train: Dict[str, Any],
    datasets_test: Dict[str, Any],
    seed: int,
) -> None:
    assert target.training_args is not None
    training_args = target.training_args
    checkpoint_dir = target.checkpoint_dir

    print(f"[{target.run_name}] Evaluating epoch {epoch} from {ckpt_path}")
    ckpt = experiment.load_checkpoint_model_state(model, ckpt_path)
    prior_complexity = ckpt.get(
        "prior_complexity",
        getattr(training_args, "prior_complexity_start", 1.0),
    )
    eval_result = experiment.run_training_epoch_eval(
        model,
        datasets_train,
        datasets_test,
        seed,
        prior_complexity,
        eval_split=target.eval_split,
    )
    log_dict = Experiment._eval_metrics_to_wandb_log_dict(
        epoch,
        val_metrics=eval_result["val_metrics"],
        val_synthetic_metrics=eval_result["val_synthetic_metrics"],
        train_eval_metrics=eval_result["train_eval_metrics"],
        test_metrics=eval_result.get("test_metrics"),
        prior_complexity=prior_complexity,
        synthetic=eval_result["synthetic"],
        eval_split=eval_result.get("eval_split", target.eval_split),
    )

    val_score = eval_result["val_score"]
    metrics_path = save_epoch_eval_metrics(
        checkpoint_dir,
        epoch,
        val_score=val_score,
        val_metrics=eval_result["val_metrics"],
        val_synthetic_metrics=eval_result["val_synthetic_metrics"],
        train_eval_metrics=eval_result["train_eval_metrics"],
        test_metrics=eval_result.get("test_metrics"),
        prior_complexity=prior_complexity,
        checkpoint_path=ckpt_path,
        eval_split=target.eval_split,
    )
    print(f"[{target.run_name}] Saved epoch {epoch} eval metrics to {metrics_path}")

    progress = _load_eval_progress(checkpoint_dir, eval_split=target.eval_split)
    progress["last_evaluated_epoch"] = epoch
    progress["eval_split"] = target.eval_split
    epoch_summaries = progress.setdefault("epoch_val_scores", {})
    epoch_summaries[str(epoch)] = val_score
    _save_eval_progress(checkpoint_dir, progress, eval_split=target.eval_split)

    # Best-checkpoint selection is val-only so a concurrent test watcher cannot
    # overwrite model_seed*_best.pt / val progress.
    if target.eval_split == "val":
        best_val_score = progress.get("best_val_score")
        if best_val_score is None or val_score > best_val_score:
            progress["best_val_score"] = val_score
            progress["best_epoch"] = epoch
            progress["best_val_metrics"] = eval_result["val_metrics"]
            best_state = deepcopy(
                (
                    model.module.state_dict()
                    if hasattr(model, "module")
                    else model.state_dict()
                )
            )
            best_path = _best_checkpoint_path(checkpoint_dir, seed)
            torch.save(best_state, best_path)
            print(
                f"[{target.run_name}] New best epoch {epoch} "
                f"(val_score={val_score:.4f}) -> {best_path}"
            )
            _save_eval_progress(
                checkpoint_dir, progress, eval_split=target.eval_split
            )

    try:
        if wandb.run is not None:
            _log_eval_to_wandb(log_dict, epoch)
    except Exception as exc:
        print(f"[{target.run_name}] WARNING: wandb upload failed for epoch {epoch}: {exc}")

    syn_acc = None
    if eval_result["val_synthetic_metrics"] is not None:
        syn_acc = eval_result["val_synthetic_metrics"]["aggregate"]["accuracy"]
    if target.eval_split == "val":
        _maybe_write_prior_complexity_override(
            checkpoint_dir,
            args=training_args,
            epoch=epoch,
            prior_complexity=prior_complexity,
            syn_acc=syn_acc,
        )

    target.progress = progress
    print(f"[{target.run_name}] Finished evaluation for epoch {epoch}", flush=True)


def _sync_dataset_cache(
    cache: DatasetCache,
    targets: List[WatchTarget],
    seed: int,
) -> None:
    prepared = [t for t in targets if t.prepared and t.training_args is not None]
    if not prepared:
        return
    needed: Set[str] = set()
    for t in prepared:
        train_names, test_names = _dataset_names_from_args(t.training_args)
        needed.update(train_names)
        needed.update(test_names)
    # Prefer first prepared run's preprocess knobs for loading.
    cache.ensure_names(needed, prepared[0].training_args, seed)


def _datasets_for_target(
    cache: DatasetCache,
    target: WatchTarget,
    seed: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    assert target.training_args is not None
    train_names, test_names = _dataset_names_from_args(target.training_args)
    cache.ensure_names(set(train_names) | set(test_names), target.training_args, seed)
    datasets_train = cache.subset(train_names)
    # Test dict: reuse train bundles when the same name appears.
    datasets_test = {}
    for name in test_names:
        if name in datasets_train:
            datasets_test[name] = datasets_train[name]
        else:
            datasets_test[name] = cache.bundles[name]
    return datasets_train, datasets_test


def watch(args: argparse.Namespace) -> None:
    checkpoint_dirs = _resolve_checkpoint_dirs(args)
    targets = [
        WatchTarget(
            checkpoint_dir=d,
            latest_only=_latest_only_for_dir(d, args),
            eval_split=_eval_split_for_dir(d, args),
        )
        for d in checkpoint_dirs
    ]

    wandb.login(key=WANDB_API_KEY)
    print(
        f"Watching {len(targets)} run(s) (round-robin, one epoch per turn): "
        + ", ".join(t.run_name for t in targets)
    )
    for t in targets:
        print(f"  [{t.run_name}] eval_split={t.eval_split}, latest_only={t.latest_only}")
    print(f"Poll interval when idle: {args.poll_interval}s")

    cache = DatasetCache()
    seen_stable: Dict[str, tuple[float, int]] = {}
    active_dir: Optional[str] = None
    experiment: Optional[Experiment] = None
    model: Optional[torch.nn.Module] = None
    rr_index = 0

    while True:
        # Lazily prepare any newly available runs.
        for target in targets:
            if target.finished or target.prepared:
                continue
            if _try_prepare_target(
                target,
                start_epoch=args.start_epoch,
                max_epoch=args.max_epoch,
                test_datasets_override=args.test_datasets,
                seed=args.seed,
            ):
                if target.finished:
                    print(f"[{target.run_name}] Already final_test_done; skipping.")

        prepared_active = [t for t in targets if t.prepared and not t.finished]
        if not any(t.prepared for t in targets):
            print("Waiting for at least one run with training_args.json...")
            time.sleep(args.poll_interval)
            continue

        if all(t.prepared and t.finished for t in targets):
            _finish_wandb_quietly()
            _release_model(model)
            print("Checkpoint watcher finished (all runs complete).")
            return

        if prepared_active:
            _sync_dataset_cache(cache, targets, args.seed)

        if not prepared_active:
            # All prepared ones finished but some dirs still lack training_args.
            time.sleep(args.poll_interval)
            continue

        did_work = False
        n = len(prepared_active)
        for _ in range(n):
            target = prepared_active[rr_index % n]
            rr_index = (rr_index % n) + 1

            pending, checkpoints = _pending_epochs(target, args.seed, seen_stable)
            if pending:
                epoch = pending[0]
                ckpt_path = checkpoints[epoch]

                # Switch W&B / model to this run if needed.
                if active_dir != target.checkpoint_dir:
                    _finish_wandb_quietly()
                    _release_model(model)
                    model = None
                    experiment = None
                    assert target.training_args is not None
                    try:
                        _init_wandb(
                            target.checkpoint_dir, target.training_args, args.seed
                        )
                    except FileNotFoundError:
                        print(
                            f"[{target.run_name}] wandb_run_id.txt missing; "
                            "evaluating without wandb until it appears."
                        )
                    experiment, model = _build_model(target.training_args)
                    active_dir = target.checkpoint_dir
                    print(
                        f"[{target.run_name}] Active model/wandb switched "
                        f"(epochs {target.effective_start}..{target.max_epoch}, "
                        f"eval_split={target.eval_split})"
                    )

                assert experiment is not None and model is not None
                datasets_train, datasets_test = _datasets_for_target(
                    cache, target, args.seed
                )
                _evaluate_one_epoch(
                    target,
                    epoch=epoch,
                    ckpt_path=ckpt_path,
                    experiment=experiment,
                    model=model,
                    datasets_train=datasets_train,
                    datasets_test=datasets_test,
                    seed=args.seed,
                )
                did_work = True
                # Fairness: one epoch then rotate.
                break

            # No pending epoch — maybe done / final test.
            if _ready_for_final(target):
                # Per-epoch test watchers already scored the test split; just finish.
                if target.eval_split != "val":
                    progress = _load_eval_progress(
                        target.checkpoint_dir, eval_split=target.eval_split
                    )
                    progress["final_test_done"] = True
                    _save_eval_progress(
                        target.checkpoint_dir,
                        progress,
                        eval_split=target.eval_split,
                    )
                    target.finished = True
                    target.progress = progress
                    print(
                        f"[{target.run_name}] Marked finished "
                        f"(eval_split={target.eval_split}; no separate final test)."
                    )
                    did_work = True
                    break

                if active_dir != target.checkpoint_dir:
                    _finish_wandb_quietly()
                    _release_model(model)
                    model = None
                    experiment = None
                    assert target.training_args is not None
                    try:
                        _init_wandb(target.checkpoint_dir, target.training_args, args.seed)
                    except FileNotFoundError:
                        print(
                            f"[{target.run_name}] wandb_run_id.txt missing for final test."
                        )
                    experiment, model = _build_model(target.training_args)
                    active_dir = target.checkpoint_dir

                assert experiment is not None and model is not None
                _, datasets_test = _datasets_for_target(cache, target, args.seed)
                progress = _load_eval_progress(
                    target.checkpoint_dir, eval_split=target.eval_split
                )
                if _run_final_test_if_ready(
                    experiment,
                    model,
                    datasets_test,
                    args.seed,
                    progress,
                    target.checkpoint_dir,
                ):
                    target.finished = True
                    target.progress = progress
                    print(f"[{target.run_name}] Marked finished after final test.")
                    did_work = True
                    break

        if all(t.prepared and t.finished for t in targets):
            _finish_wandb_quietly()
            _release_model(model)
            print("Checkpoint watcher finished (all runs complete).")
            return

        if not did_work:
            time.sleep(args.poll_interval)


def main() -> None:
    watch(_parse_args())


if __name__ == "__main__":
    main()
