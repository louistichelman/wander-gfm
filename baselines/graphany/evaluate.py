"""In-process GraphAny inference on a raw npy bundle (released checkpoint).

The bundle stores ``load_base_dataset`` tensors plus Wander protocol
``split.npz`` / ``split_{i}.npz``. GraphAny applies its own bidirect /
LinearGNN / entropy-distance preprocess. Inference uses the released
checkpoint (no training).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from data.nc_splits import n_eval_nc_splits, split_npz_filename


CKPT_FILES = {
    "arxiv": "graph_any_arxiv.pt",
    "cora": "graph_any_cora.pt",
    "wisconsin": "graph_any_wisconsin.pt",
    "product": "graph_any_product.pt",
}


def resolve_ckpt_path(graphany_root: Path, ckpt: str) -> Path:
    key = ckpt.strip().lower()
    name = CKPT_FILES.get(key, key)
    if not name.endswith(".pt"):
        name = f"graph_any_{name}.pt"
    path = Path(name)
    if not path.is_absolute():
        path = graphany_root / "checkpoints" / path.name
    if not path.is_file():
        raise FileNotFoundError(f"GraphAny checkpoint not found: {path}")
    return path


def dump_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


def _softmax(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits, dim=-1).detach().cpu().numpy()


def _auroc(y: np.ndarray, probs: np.ndarray) -> float | None:
    n_class = int(probs.shape[1])
    try:
        if n_class == 2:
            return float(roc_auc_score(y, probs[:, 1]))
        return float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))
    except ValueError:
        return None


def _score_split(model, ds, mask: torch.Tensor, device: torch.device) -> dict[str, Any]:
    idx = mask.nonzero(as_tuple=False).view(-1)
    if idx.numel() == 0:
        return {"accuracy": None, "roc-auc": None, "n": 0}
    idx = idx.to(device)
    ds.to(device)
    processed = {c: t.to(device) for c, t in ds.unmasked_pred.items()}
    dist = ds.dist.to(device)[idx] if ds.dist is not None else None
    logits, _attn = model(
        {c: processed[c][idx] for c in model.feat_channels},
        dist=dist,
    )
    y = ds.label[idx].detach().cpu().numpy()
    pred = logits.argmax(dim=-1).detach().cpu().numpy()
    probs = _softmax(logits)
    acc = float(accuracy_score(y, pred))
    return {"accuracy": acc, "roc-auc": _auroc(y, probs), "n": int(idx.numel())}


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _select_device(cpu: bool) -> torch.device:
    """Pick inference device; fall back when this wheel cannot run the GPU."""
    if cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    major, minor = torch.cuda.get_device_capability(0)
    # graphany_env is torch 2.4+cu124 (sm_50..sm_90); newer GPUs need a newer wheel.
    if major >= 12:
        print(
            f"GPU sm_{major}{minor} is not supported by this PyTorch/DGL wheel; "
            "using CPU."
        )
        return torch.device("cpu")
    return torch.device("cuda")


def evaluate_slug(
    *,
    graphany_root: Path,
    slug: str,
    registry_key: str,
    data_root: Path,
    ckpt: str,
    cache_dir: Path,
    split_index: int = 0,
    device: torch.device | None = None,
    cpu: bool = False,
) -> dict[str, Any]:
    """Run GraphAny ICL-style inference on one Wander protocol split."""
    graphany_root = graphany_root.resolve()
    if str(graphany_root) not in sys.path:
        sys.path.insert(0, str(graphany_root))

    os.environ["GRAPHANY_DATA_ROOT"] = str(data_root.resolve())

    from omegaconf import OmegaConf

    from graphany.data import GraphDataset
    from graphany.model import GraphAny

    ckpt_path = resolve_ckpt_path(graphany_root, ckpt)
    if device is None:
        device = _select_device(cpu)

    n_protocol = n_eval_nc_splits(registry_key)
    split_file = split_npz_filename(registry_key, split_index, n_protocol)
    split_path = data_root / slug / split_file
    if not split_path.is_file():
        raise FileNotFoundError(
            f"Missing {split_path}. Re-run: python baselines/graphany/prepare.py "
            f"--dataset {registry_key}"
        )

    blob = _torch_load(ckpt_path)
    graph_any_config = dict(blob["graph_any_config"])
    feat_channels = list(graph_any_config["feat_channels"])
    pred_channels = list(graph_any_config["pred_channels"])
    entropy = graph_any_config.get("entropy", 1)

    cfg = OmegaConf.create(
        {
            "_ds_meta_data": {},
            "dirs": {"data_storage": str(data_root.resolve()) + "/"},
            "wander_data_root": str(data_root.resolve()),
            "wander_split_file": split_file,
            "wander_split_index": int(split_index),
            "feat_chn": "+".join(feat_channels),
            "pred_chn": "+".join(pred_channels),
            "feat_channels": feat_channels,
            "pred_channels": pred_channels,
            "n_hops": 2,
            "add_self_loop": False,
            "to_bidirected": True,
            "entropy": entropy,
            "seed": int(split_index),
            "n_per_label_examples": 5,
        }
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    for stale in cache_dir.glob("*.pt"):
        if stale.stat().st_size < 32:
            stale.unlink()
    # DGL 2.4 CUDA kernels may be missing on newer GPU archs; preprocess on CPU.
    preprocess_device = torch.device("cpu")
    ds = GraphDataset(
        cfg,
        slug,
        str(cache_dir),
        n_hops=cfg.n_hops,
        preprocess_device=preprocess_device,
    )

    model = GraphAny(**graph_any_config)
    state = blob["state_dict"]
    stripped = {
        key[len("gnn_model.") :]: value
        for key, value in state.items()
        if key.startswith("gnn_model.")
    }
    model.load_state_dict(stripped, strict=True)
    model.to(device)
    model.eval()

    with torch.no_grad():
        val_metrics = _score_split(model, ds, ds.val_mask, device)
        test_metrics = _score_split(model, ds, ds.test_mask, device)

    return {
        "function": "baselines.graphany.evaluate.evaluate_slug",
        "slug": slug,
        "registry_key": registry_key,
        "split_index": int(split_index),
        "split_file": split_file,
        "checkpoint": str(ckpt_path),
        "n_nodes": int(ds.n_nodes),
        "n_classes": int(ds.num_class),
        "metrics": {"val": val_metrics, "test": test_metrics},
        "status": "ok",
    }
