"""Utility functions for experiments."""

import argparse
import json
import os
import random
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import torch
import string


def set_seed(seed):
    """Set random seed for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False


def generate_experiment_name(args) -> str:
    """Generate experiment name from key arguments and timestamp.
    
    Args:
        args: Parsed command-line arguments.
        
    Returns:
        Experiment name string in format:
        {datasets}_{task}_{gnn_type}_L{num_layers}_lr{lr}_{loss_type}_{datetime}
    """
    random_string = ''
    
    # Timestamp (short format: MMDD_HHMM)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    
    return f"{timestamp}_{''.join(random.choices(string.ascii_letters + string.digits, k=2))}"


def get_wandb_run_name(base_name: str, seed: int, mode: str) -> str:
    """Generate W&B run name.
    
    Args:
        base_name: Base experiment name from generate_experiment_name().
        seed: Random seed (ignored for summary mode).
        mode: "train", "eval", or "summary".
        
    Returns:
        W&B run name string.
    """
    if mode == "summary":
        return f"summary_{base_name}"
    return f"{mode}_{base_name}_seed{seed}"


def get_checkpoint_dir(save_load_path: str, base_name: str) -> str:
    """Get checkpoint directory for this experiment.
    
    Args:
        save_load_path: Root checkpoint directory.
        base_name: Base experiment name.
        
    Returns:
        Full path to experiment-specific checkpoint directory.
    """
    return os.path.join(save_load_path, base_name)


TRAINING_ARGS_FILENAME = "training_args.json"
EVAL_ARGS_FILENAME = "eval_args.json"
EVAL_ARGS_TXT_FILENAME = "eval_args.txt"
WANDB_RUN_ID_FILENAME = "wandb_run_id.txt"
TRAINING_STATUS_FILENAME = "training_status.json"
PRIOR_COMPLEXITY_FILENAME = "prior_complexity.json"
EVAL_PROGRESS_FILENAME = "eval_progress.json"
EVAL_EPOCHS_DIR = "eval_epochs"
FINAL_TEST_METRICS_FILENAME = "final_test.json"
BEST_VAL_METRICS_FILENAME = "best_val.json"


def eval_epochs_dir(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, EVAL_EPOCHS_DIR)


def epoch_eval_metrics_path(
    checkpoint_dir: str, epoch: int, eval_split: str = "val"
) -> str:
    """Path for per-epoch watcher metrics; non-val splits get a suffix."""
    if eval_split == "val":
        name = f"epoch_{epoch:04d}.json"
    else:
        name = f"epoch_{epoch:04d}_{eval_split}.json"
    return os.path.join(eval_epochs_dir(checkpoint_dir), name)


def final_test_metrics_path(
    checkpoint_dir: str, filename: str = FINAL_TEST_METRICS_FILENAME
) -> str:
    return os.path.join(eval_epochs_dir(checkpoint_dir), filename)


def save_epoch_eval_metrics(
    checkpoint_dir: str,
    epoch: int,
    *,
    val_score: float,
    val_metrics: Optional[Dict[str, Any]],
    val_synthetic_metrics: Optional[Dict[str, Any]] = None,
    train_eval_metrics: Optional[Dict[str, Any]] = None,
    test_metrics: Optional[Dict[str, Any]] = None,
    prior_complexity: Optional[float] = None,
    checkpoint_path: Optional[str] = None,
    eval_split: str = "val",
) -> str:
    """Persist full validation metrics for one epoch under eval_epochs/."""
    out_dir = eval_epochs_dir(checkpoint_dir)
    os.makedirs(out_dir, exist_ok=True)
    path = epoch_eval_metrics_path(checkpoint_dir, epoch, eval_split=eval_split)
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "eval_split": eval_split,
        "val_score": val_score,
        "val_metrics": val_metrics or {},
    }
    if checkpoint_path is not None:
        payload["checkpoint"] = os.path.basename(checkpoint_path)
    if prior_complexity is not None:
        payload["prior_complexity"] = prior_complexity
    if val_synthetic_metrics is not None:
        payload["val_synthetic_metrics"] = val_synthetic_metrics
    if train_eval_metrics is not None:
        payload["train_eval_metrics"] = train_eval_metrics
    if test_metrics is not None:
        payload["test_metrics"] = test_metrics
    save_json(path, payload)
    return path


def save_best_val_metrics(
    checkpoint_dir: str,
    *,
    best_epoch: int,
    best_val_score: float,
    val_metrics: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    filename: str = BEST_VAL_METRICS_FILENAME,
) -> str:
    """Persist the best validation score used for checkpoint / LR selection."""
    path = os.path.join(checkpoint_dir, filename)
    payload: Dict[str, Any] = {
        "best_epoch": best_epoch,
        "best_val_score": float(best_val_score),
        "val_metrics": val_metrics or {},
    }
    if extra:
        payload.update(extra)
    save_json(path, payload)
    return path


def save_final_test_metrics(
    checkpoint_dir: str,
    *,
    best_epoch: int,
    test_metrics: Dict[str, Any],
    filename: str = FINAL_TEST_METRICS_FILENAME,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist final test-split metrics for the best validation checkpoint."""
    out_dir = eval_epochs_dir(checkpoint_dir)
    os.makedirs(out_dir, exist_ok=True)
    path = final_test_metrics_path(checkpoint_dir, filename=filename)
    payload: Dict[str, Any] = {
        "best_epoch": best_epoch,
        "test_metrics": test_metrics,
    }
    if extra:
        payload.update(extra)
    save_json(path, payload)
    return path


def namespace_from_dict(data: Dict[str, Any]) -> argparse.Namespace:
    """Reconstruct an argparse Namespace from a JSON-compatible dict."""
    return argparse.Namespace(**data)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_training_args(checkpoint_dir: str, args) -> None:
    save_json(os.path.join(checkpoint_dir, TRAINING_ARGS_FILENAME), vars(args))


def _json_safe_args(args) -> Dict[str, Any]:
    """Convert argparse Namespace to a JSON-serializable dict."""
    payload: Dict[str, Any] = {}
    for key, value in sorted(vars(args).items()):
        if isinstance(value, (str, int, float, bool)) or value is None:
            payload[key] = value
        elif isinstance(value, (list, tuple)):
            payload[key] = list(value)
        elif isinstance(value, dict):
            payload[key] = value
        else:
            payload[key] = str(value)
    return payload


def save_eval_run_args(save_load_path: str, args) -> str:
    """Persist CLI keyword arguments for an eval-only run (first write wins)."""
    os.makedirs(save_load_path, exist_ok=True)
    json_path = os.path.join(save_load_path, EVAL_ARGS_FILENAME)
    if os.path.exists(json_path):
        return json_path

    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "arguments": _json_safe_args(args),
    }
    save_json(json_path, payload)

    txt_path = os.path.join(save_load_path, EVAL_ARGS_TXT_FILENAME)
    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write(f"saved_at: {payload['saved_at']}\n")
        for key, value in payload["arguments"].items():
            handle.write(f"{key}={value}\n")
        handle.write("\n")
    return json_path


def save_wandb_run_id(checkpoint_dir: str, run_id: str) -> None:
    with open(os.path.join(checkpoint_dir, WANDB_RUN_ID_FILENAME), "w", encoding="utf-8") as handle:
        handle.write(run_id)
        handle.write("\n")


def load_wandb_run_id(checkpoint_dir: str) -> str:
    with open(os.path.join(checkpoint_dir, WANDB_RUN_ID_FILENAME), encoding="utf-8") as handle:
        return handle.read().strip()


def write_training_status(
    checkpoint_dir: str,
    *,
    status: str,
    last_completed_epoch: int,
    max_epochs: int,
) -> None:
    save_json(
        os.path.join(checkpoint_dir, TRAINING_STATUS_FILENAME),
        {
            "status": status,
            "last_completed_epoch": last_completed_epoch,
            "max_epochs": max_epochs,
        },
    )


def read_prior_complexity_override(checkpoint_dir: str, epoch: int) -> Optional[float]:
    path = os.path.join(checkpoint_dir, PRIOR_COMPLEXITY_FILENAME)
    if not os.path.isfile(path):
        return None
    payload = load_json(path)
    apply_from = int(payload.get("apply_from_epoch", -1))
    if apply_from > epoch:
        return None
    return float(payload["prior_complexity"])


def write_prior_complexity_override(
    checkpoint_dir: str,
    *,
    apply_from_epoch: int,
    prior_complexity: float,
    based_on_epoch: int,
    syn_acc: float,
) -> None:
    save_json(
        os.path.join(checkpoint_dir, PRIOR_COMPLEXITY_FILENAME),
        {
            "apply_from_epoch": apply_from_epoch,
            "prior_complexity": prior_complexity,
            "based_on_epoch": based_on_epoch,
            "syn_acc": syn_acc,
        },
    )


def define_async_wandb_metrics() -> None:
    """Configure separate x-axes for async train vs eval metrics on the same run."""
    import wandb

    wandb.define_metric("train_epoch")
    wandb.define_metric("train/*", step_metric="train_epoch")
    wandb.define_metric("prior_complexity_active", step_metric="train_epoch")

    wandb.define_metric("eval_epoch")
    wandb.define_metric("val/*", step_metric="eval_epoch")
    wandb.define_metric("test/*", step_metric="eval_epoch")
    wandb.define_metric("val_synthetic/*", step_metric="eval_epoch")
    wandb.define_metric("train_eval/*", step_metric="eval_epoch")
    wandb.define_metric("coverage/val/*", step_metric="eval_epoch")
    wandb.define_metric("coverage/test/*", step_metric="eval_epoch")
    wandb.define_metric("coverage/val_synthetic/*", step_metric="eval_epoch")
    wandb.define_metric("coverage/train_eval/*", step_metric="eval_epoch")
    wandb.define_metric("prior_complexity", step_metric="eval_epoch")
    wandb.define_metric("prior_complexity_next", step_metric="eval_epoch")
    wandb.define_metric("prior_complexity_apply_from_epoch", step_metric="eval_epoch")


def get_checkpoint_path(save_load_path: str, base_name: str, seed: int, epoch=None) -> str:
    """Get full checkpoint path for a specific seed and epoch.
    
    Args:
        save_load_path: Root checkpoint directory.
        base_name: Base experiment name.
        seed: Random seed.
        epoch: None or "best" for the best-model checkpoint,
               or an int for an epoch-specific checkpoint.
        
    Returns:
        Full path to model checkpoint file.
    """
    if epoch is None or epoch == "best":
        filename = f"model_seed{seed}_best.pt"
    else:
        filename = f"model_seed{seed}_epoch{int(epoch)}.pt"
    return os.path.join(save_load_path, base_name, filename)


def capture_rng_state() -> dict:
    """Snapshot the RNGs that drive synthetic dataset generation."""
    state = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    """Restore a snapshot from ``capture_rng_state``."""
    random.setstate(state["random"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def seed_rngs(seed: int) -> None:
    """Seed only the RNGs, leaving backend flags alone.

    ``set_seed`` also disables cuDNN and forces deterministic kernels, which is
    fine once at startup but must not be triggered mid-run (it would persist
    after the block that wanted reproducible draws and slow down training).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
