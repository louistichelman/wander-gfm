"""Training loop for the NBFNet LP baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import torch
from torch import nn

from .data import (
    NBFNetGraph,
    iter_triple_batches,
    remove_easy_edges,
    sample_negative_tails,
)
from baselines.lp_loss import pairwise_bce_loss

from .eval import evaluate_on_split
from .model import NBFNet

PathLike = Union[str, Path]


@dataclass
class TrainConfig:
    input_dim: int = 32
    hidden_dims: tuple = (32, 32, 32, 32, 32, 32)
    num_negative: int = 32
    equal_pos_neg_weight: bool = False
    batch_size: int = 16
    eval_batch_size: int = 4
    batches_per_epoch: Optional[int] = None
    lr: float = 5e-3
    weight_decay: float = 0.0
    max_epochs: int = 40
    patience: int = 5
    use_node_features: bool = True
    max_eval_queries: Optional[int] = None
    max_monitor_queries: Optional[int] = None
    eval_every: int = 1
    seed: int = 0
    device: str = "cuda"
    recall_k: int = 20


def _save_best_checkpoint(
    path: Path,
    *,
    state_dict: Dict[str, torch.Tensor],
    cfg: TrainConfig,
    graph: NBFNetGraph,
    use_node_features: bool,
    best_epoch: int,
    best_metric: float,
    metric_key: str,
    last_completed_epoch: int,
    epochs_no_improve: int,
    history: Optional[list] = None,
) -> None:
    """Atomic overwrite of the single best/resume checkpoint.

    ``state_dict`` must be the **best** weights (CPU tensors OK).
    ``last_completed_epoch`` / ``epochs_no_improve`` track progress for resume.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in state_dict.items()},
        "best_epoch": best_epoch,
        "best_val_metric": best_metric,
        "metric_key": metric_key,
        "last_completed_epoch": last_completed_epoch,
        "epochs_no_improve": epochs_no_improve,
        "history": list(history or []),
        "dataset": graph.bundle.name,
        "num_nodes": graph.num_nodes,
        "num_relations": graph.num_relations,
        "use_node_features": use_node_features,
        "input_dim": cfg.input_dim,
        "hidden_dims": list(cfg.hidden_dims),
        "node_feature_dim": (
            int(graph.x.shape[1]) if use_node_features and graph.x is not None else None
        ),
        "seed": cfg.seed,
        "max_epochs": cfg.max_epochs,
        "patience": cfg.patience,
    }
    torch.save(payload, tmp)
    tmp.replace(path)


def _load_resume_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    log: Callable[[str], None],
) -> Dict:
    """Load ``best.pt``, restore best weights, return resume bookkeeping."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"])
    best_epoch = int(payload.get("best_epoch", -1))
    best_metric = float(payload.get("best_val_metric", -1.0))
    last_completed = int(payload.get("last_completed_epoch", best_epoch))
    # Continue after the furthest completed epoch; weights are the best so far.
    start_epoch = max(last_completed, best_epoch) + 1
    epochs_no_improve = int(payload.get("epochs_no_improve", 0))
    history = list(payload.get("history") or [])
    log(
        f"[nbfnet] resume from {path}: best_epoch={best_epoch} "
        f"best_metric={best_metric:.4f} last_completed={last_completed} "
        f"-> start_epoch={start_epoch} epochs_no_improve={epochs_no_improve}"
    )
    return {
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "start_epoch": start_epoch,
        "epochs_no_improve": epochs_no_improve,
        "history": history,
        "best_state_cpu": {k: v.detach().cpu().clone() for k, v in payload["state_dict"].items()},
    }


def _resolve_use_features(graph: NBFNetGraph, flag: bool) -> bool:
    if not flag:
        return False
    return graph.x is not None


def build_model(graph: NBFNetGraph, cfg: TrainConfig) -> NBFNet:
    use_feats = _resolve_use_features(graph, cfg.use_node_features)
    feat_dim = int(graph.x.shape[1]) if use_feats and graph.x is not None else None
    return NBFNet(
        input_dim=cfg.input_dim,
        num_relation=graph.num_relations,
        hidden_dims=cfg.hidden_dims,
        use_node_features=use_feats,
        node_feature_dim=feat_dim,
    )


def _bce_loss(
    logits: torch.Tensor, *, equal_pos_neg_weight: bool = False
) -> torch.Tensor:
    return pairwise_bce_loss(logits, equal_pos_neg_weight=equal_pos_neg_weight)


def train_nbfnet(
    graph: NBFNetGraph,
    cfg: TrainConfig,
    *,
    log_fn: Optional[Callable[[str], None]] = None,
    checkpoint_path: Optional[PathLike] = None,
    resume: bool = True,
) -> Dict:
    """Train NBFNet; return metrics dict including best val / final test.

    If ``checkpoint_path`` is set, the running best weights + progress are written
    there each epoch (atomic overwrite). When ``resume`` is True and the file
    exists, training continues from ``last_completed_epoch + 1`` with the stored
    best weights (preemption / Slurm requeue safe).
    """
    log = log_fn or (lambda s: print(s, flush=True))
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
        log("[nbfnet] CUDA unavailable; using CPU")

    ckpt_path = Path(checkpoint_path) if checkpoint_path is not None else None

    torch.manual_seed(cfg.seed)
    model = build_model(graph, cfg).to(device)
    use_feats = model.use_node_features
    x = graph.x.to(device) if use_feats and graph.x is not None else None

    # Keep graph edges on device
    graph.edge_index = graph.edge_index.to(device)
    graph.edge_type = graph.edge_type.to(device)
    graph.train_triples = graph.train_triples.to(device)
    graph.val_triples = graph.val_triples.to(device)
    graph.test_triples = graph.test_triples.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(cfg.seed)

    primary = graph.preferred_link_pred_eval
    preferred_split = str(graph.bundle.metadata.get("preferred_eval_split") or "val")
    monitor_split = "test" if preferred_split == "test" else "val"
    if monitor_split == "val" and int(graph.val_triples.shape[0]) == 0:
        monitor_split = "test"
    metric_key = "mrr" if primary == "mrr" else f"recall@{cfg.recall_k}"
    best_metric = -1.0
    best_epoch = -1
    best_state_cpu: Optional[Dict[str, torch.Tensor]] = None
    epochs_no_improve = 0
    history: list = []
    start_epoch = 1
    resumed = False

    if resume and ckpt_path is not None and ckpt_path.is_file():
        meta = _load_resume_checkpoint(ckpt_path, model, log=log)
        best_epoch = meta["best_epoch"]
        best_metric = meta["best_metric"]
        start_epoch = meta["start_epoch"]
        epochs_no_improve = meta["epochs_no_improve"]
        history = meta["history"]
        best_state_cpu = meta["best_state_cpu"]
        resumed = True
        # Re-seed so resumed runs don't replay the same RNG stream from epoch 1.
        gen.manual_seed(cfg.seed + max(start_epoch, 1) * 100_003)
        model.to(device)

    n_train = int(graph.train_triples.shape[0])
    bpe = cfg.batches_per_epoch
    use_capped_epoch = (
        bpe is not None and bpe > 0 and n_train > int(bpe) * int(cfg.batch_size)
    )
    max_batches = int(bpe) if use_capped_epoch else None

    log(
        f"[nbfnet] train {graph.bundle.name}: N={graph.num_nodes} R={graph.num_relations} "
        f"train_q={n_train} val_q={graph.val_triples.shape[0]} "
        f"test_q={graph.test_triples.shape[0]} feats={use_feats} protocol={primary} "
        f"monitor={monitor_split} num_neg={cfg.num_negative} "
        f"equal_pos_neg={int(cfg.equal_pos_neg_weight)}"
        + (
            f" batches_per_epoch={max_batches} (capped)"
            if use_capped_epoch
            else " batches_per_epoch=full"
        )
        + (f" ckpt={ckpt_path}" if ckpt_path is not None else "")
        + (f" resume_from_epoch={start_epoch}" if resumed else "")
    )

    if start_epoch > cfg.max_epochs:
        log(
            f"[nbfnet] checkpoint already finished "
            f"(last/best past max_epochs={cfg.max_epochs}); skipping train loop"
        )
    elif epochs_no_improve >= cfg.patience and best_epoch > 0:
        log(
            f"[nbfnet] checkpoint already early-stopped "
            f"(epochs_no_improve={epochs_no_improve} >= patience={cfg.patience}); "
            "skipping train loop"
        )
        start_epoch = cfg.max_epochs + 1

    monitor_queries = (
        cfg.max_eval_queries if cfg.max_eval_queries is not None else cfg.max_monitor_queries
    )
    final_queries = cfg.max_eval_queries

    for epoch in range(start_epoch, cfg.max_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for pos in iter_triple_batches(
            graph.train_triples,
            cfg.batch_size,
            shuffle=True,
            generator=gen,
            max_batches=max_batches,
        ):
            pos = pos.to(device)
            batch = sample_negative_tails(
                graph.num_nodes, pos, cfg.num_negative, generator=gen
            )
            ei, et = remove_easy_edges(graph, batch[:, 0, 0], batch[:, 0, 1], batch[:, 0, 2])
            logits = model(ei, et, graph.num_nodes, batch, node_features=x)
            loss = _bce_loss(
                logits, equal_pos_neg_weight=cfg.equal_pos_neg_weight
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        row = {"epoch": epoch, "train_loss": avg_loss, "num_batches": n_batches}
        stop_early = False

        if epoch % cfg.eval_every == 0 or epoch == cfg.max_epochs:
            val_metrics = evaluate_on_split(
                model,
                graph,
                split=monitor_split,
                max_queries=monitor_queries,
                seed=cfg.seed,
                recall_k=cfg.recall_k,
                device=device,
                node_features=x,
                batch_size=cfg.eval_batch_size,
            )
            row.update({f"val/{k}": v for k, v in val_metrics.items()})
            cur = float(val_metrics.get(metric_key, 0.0))
            log(
                f"[nbfnet] epoch {epoch}/{cfg.max_epochs} loss={avg_loss:.4f} "
                + " ".join(f"{monitor_split}_{k}={v:.4f}" for k, v in val_metrics.items())
            )
            if cur > best_metric + 1e-6:
                best_metric = cur
                best_epoch = epoch
                epochs_no_improve = 0
                best_state_cpu = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                log(
                    f"[nbfnet] new best {metric_key}={best_metric:.4f} @ epoch {best_epoch}"
                )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= cfg.patience:
                    stop_early = True
                    log(
                        f"[nbfnet] early stop at epoch {epoch} "
                        f"(best {metric_key}={best_metric:.4f} @ {best_epoch})"
                    )
        else:
            log(f"[nbfnet] epoch {epoch}/{cfg.max_epochs} loss={avg_loss:.4f}")

        history.append(row)

        # Persist best weights + progress every epoch (preemption-safe).
        if ckpt_path is not None and best_state_cpu is not None:
            _save_best_checkpoint(
                ckpt_path,
                state_dict=best_state_cpu,
                cfg=cfg,
                graph=graph,
                use_node_features=use_feats,
                best_epoch=best_epoch,
                best_metric=best_metric,
                metric_key=metric_key,
                last_completed_epoch=epoch,
                epochs_no_improve=epochs_no_improve,
                history=history,
            )
        elif ckpt_path is not None and best_state_cpu is None:
            # No val improvement yet: still snapshot current weights as provisional best.
            provisional = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            _save_best_checkpoint(
                ckpt_path,
                state_dict=provisional,
                cfg=cfg,
                graph=graph,
                use_node_features=use_feats,
                best_epoch=best_epoch if best_epoch > 0 else epoch,
                best_metric=best_metric,
                metric_key=metric_key,
                last_completed_epoch=epoch,
                epochs_no_improve=epochs_no_improve,
                history=history,
            )
            if best_epoch <= 0:
                best_epoch = epoch
                best_state_cpu = provisional

        if stop_early:
            break

    # Restore best weights: prefer in-memory best, else on-disk checkpoint.
    if best_state_cpu is not None:
        model.load_state_dict(best_state_cpu)
    elif ckpt_path is not None and ckpt_path.is_file():
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state_dict"])
        best_epoch = int(payload.get("best_epoch", best_epoch))
        best_metric = float(payload.get("best_val_metric", best_metric))

    test_metrics = evaluate_on_split(
        model,
        graph,
        split="test",
        max_queries=final_queries,
        seed=cfg.seed,
        recall_k=cfg.recall_k,
        device=device,
        node_features=x,
        batch_size=cfg.eval_batch_size,
    )
    log(
        f"[nbfnet] test @"
        + " ".join(f"{k}={v:.4f}" for k, v in test_metrics.items())
        + f" (best_epoch={best_epoch})"
    )

    val_metrics = evaluate_on_split(
        model,
        graph,
        split=monitor_split,
        max_queries=monitor_queries,
        seed=cfg.seed,
        recall_k=cfg.recall_k,
        device=device,
        node_features=x,
        batch_size=cfg.eval_batch_size,
    )
    out = {
        "best_epoch": best_epoch,
        "best_val_metric": best_metric,
        "preferred_link_pred_eval": primary,
        "monitor_split": monitor_split,
        "use_node_features": use_feats,
        "num_nodes": graph.num_nodes,
        "num_relations": graph.num_relations,
        "num_train_triples": int(graph.train_triples.shape[0]),
        "num_val_triples": int(graph.val_triples.shape[0]),
        "num_test_triples": int(graph.test_triples.shape[0]),
        "history": history,
        "resumed": resumed,
        **{f"test_{k}": v for k, v in test_metrics.items()},
        **{f"val_{k}": v for k, v in val_metrics.items()},
    }
    if ckpt_path is not None:
        out["checkpoint_path"] = str(ckpt_path)
    return out
