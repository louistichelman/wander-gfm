"""Training loop for the BUDDY LP baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import torch
from torch import nn

from .data import (
    BuddyGraph,
    buddy_to_device,
    iter_triple_batches,
    prepare_buddy_graph,
    sample_negative_dst,
)
from baselines.lp_loss import pairwise_bce_loss

from .eval import evaluate_on_split, score_pairs
from .hashing import HashConfig
from .model import BUDDY
from baselines.nbfnet.data import NBFNetGraph

PathLike = Union[str, Path]


@dataclass
class TrainConfig:
    hidden_channels: int = 256
    max_hash_hops: int = 2
    minhash_num_perm: int = 128
    hll_p: int = 8
    use_zero_one: bool = False
    floor_sf: bool = False
    add_normed_features: bool = False
    sign_k: int = 0
    sign_dropout: float = 0.5
    label_dropout: float = 0.5
    feature_dropout: float = 0.5
    use_node_features: bool = True
    train_node_embedding: bool = False
    propagate_embeddings: bool = False
    num_negative: int = 32
    equal_pos_neg_weight: bool = False
    batch_size: int = 1024
    eval_pair_batch_size: int = 250_000
    batches_per_epoch: Optional[int] = None
    lr: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 100
    patience: int = 10
    max_eval_queries: Optional[int] = None
    max_monitor_queries: Optional[int] = 512
    eval_every: int = 1
    seed: int = 0
    device: str = "cuda"
    recall_k: int = 20


def hash_config_from_train(cfg: TrainConfig) -> HashConfig:
    return HashConfig(
        max_hash_hops=cfg.max_hash_hops,
        minhash_num_perm=cfg.minhash_num_perm,
        hll_p=cfg.hll_p,
        use_zero_one=cfg.use_zero_one,
        floor_sf=cfg.floor_sf,
    )


def build_model(state: BuddyGraph, cfg: TrainConfig) -> BUDDY:
    use_emb = state.use_embedding or cfg.train_node_embedding
    return BUDDY(
        num_struct_features=state.hasher.num_struct_features(),
        hidden_channels=cfg.hidden_channels,
        num_features=state.raw_feature_dim,
        use_feature=state.use_feature,
        sign_k=cfg.sign_k,
        sign_dropout=cfg.sign_dropout,
        label_dropout=cfg.label_dropout,
        feature_dropout=cfg.feature_dropout,
        add_normed_features=cfg.add_normed_features,
        use_embedding=use_emb,
        propagate_embeddings=cfg.propagate_embeddings and use_emb,
    )


def _save_best_checkpoint(
    path: Path,
    *,
    state_dict: Dict[str, torch.Tensor],
    embedding_state: Optional[Dict[str, torch.Tensor]],
    cfg: TrainConfig,
    state: BuddyGraph,
    best_epoch: int,
    best_metric: float,
    metric_key: str,
    last_completed_epoch: int,
    epochs_no_improve: int,
    history: Optional[list] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in state_dict.items()},
        "embedding_state": (
            {k: v.detach().cpu() for k, v in embedding_state.items()} if embedding_state else None
        ),
        "best_epoch": best_epoch,
        "best_val_metric": best_metric,
        "metric_key": metric_key,
        "last_completed_epoch": last_completed_epoch,
        "epochs_no_improve": epochs_no_improve,
        "history": list(history or []),
        "dataset": state.graph.bundle.name,
        "num_nodes": state.graph.num_nodes,
        "use_feature": state.use_feature,
        "use_embedding": state.use_embedding,
        "hidden_channels": cfg.hidden_channels,
        "sign_k": cfg.sign_k,
        "seed": cfg.seed,
        "max_epochs": cfg.max_epochs,
        "patience": cfg.patience,
    }
    torch.save(payload, tmp)
    tmp.replace(path)


def _load_resume_checkpoint(
    path: Path,
    model: nn.Module,
    embedding: Optional[nn.Embedding],
    *,
    log: Callable[[str], None],
) -> Dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"])
    if embedding is not None and payload.get("embedding_state") is not None:
        embedding.load_state_dict(payload["embedding_state"])
    best_epoch = int(payload.get("best_epoch", -1))
    best_metric = float(payload.get("best_val_metric", -1.0))
    last_completed = int(payload.get("last_completed_epoch", best_epoch))
    start_epoch = max(last_completed, best_epoch) + 1
    epochs_no_improve = int(payload.get("epochs_no_improve", 0))
    history = list(payload.get("history") or [])
    log(
        f"[buddy] resume from {path}: best_epoch={best_epoch} "
        f"best_metric={best_metric:.4f} last_completed={last_completed} "
        f"-> start_epoch={start_epoch} epochs_no_improve={epochs_no_improve}"
    )
    best_state = {k: v.detach().cpu().clone() for k, v in payload["state_dict"].items()}
    best_emb = None
    if payload.get("embedding_state") is not None:
        best_emb = {k: v.detach().cpu().clone() for k, v in payload["embedding_state"].items()}
    return {
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "start_epoch": start_epoch,
        "epochs_no_improve": epochs_no_improve,
        "history": history,
        "best_state_cpu": best_state,
        "best_emb_cpu": best_emb,
    }


def _snapshot(model: nn.Module, embedding: Optional[nn.Embedding]):
    weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    emb = None
    if embedding is not None:
        emb = {k: v.detach().cpu().clone() for k, v in embedding.state_dict().items()}
    return weights, emb


def train_buddy(
    graph: NBFNetGraph,
    cfg: TrainConfig,
    *,
    log_fn: Optional[Callable[[str], None]] = None,
    checkpoint_path: Optional[PathLike] = None,
    resume: bool = True,
) -> Dict:
    """Train BUDDY; return metrics including best val / final test."""
    log = log_fn or (lambda s: print(s, flush=True))
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
        log("[buddy] CUDA unavailable; using CPU")

    ckpt_path = Path(checkpoint_path) if checkpoint_path is not None else None
    torch.manual_seed(cfg.seed)

    log(f"[buddy] building hashes on {graph.bundle.name} N={graph.num_nodes} E={graph.edge_index.size(1)}")
    state = prepare_buddy_graph(
        graph,
        hash_cfg=hash_config_from_train(cfg),
        use_node_features=cfg.use_node_features,
        sign_k=cfg.sign_k,
        device=device,
    )
    state = buddy_to_device(state, device)
    model = build_model(state, cfg).to(device)

    embedding: Optional[nn.Embedding] = None
    if model.use_embedding:
        embedding = nn.Embedding(graph.num_nodes, cfg.hidden_channels).to(device)
        nn.init.xavier_uniform_(embedding.weight)

    params = list(model.parameters())
    if embedding is not None:
        params += list(embedding.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(cfg.seed)

    metric_key = f"recall@{cfg.recall_k}"
    best_metric = -1.0
    best_epoch = -1
    best_state_cpu: Optional[Dict[str, torch.Tensor]] = None
    best_emb_cpu: Optional[Dict[str, torch.Tensor]] = None
    epochs_no_improve = 0
    history: list = []
    start_epoch = 1
    resumed = False

    if resume and ckpt_path is not None and ckpt_path.is_file():
        meta = _load_resume_checkpoint(ckpt_path, model, embedding, log=log)
        best_epoch = meta["best_epoch"]
        best_metric = meta["best_metric"]
        start_epoch = meta["start_epoch"]
        epochs_no_improve = meta["epochs_no_improve"]
        history = meta["history"]
        best_state_cpu = meta["best_state_cpu"]
        best_emb_cpu = meta["best_emb_cpu"]
        resumed = True
        gen.manual_seed(cfg.seed + max(start_epoch, 1) * 100_003)
        model.to(device)
        if embedding is not None:
            embedding.to(device)

    n_train = int(graph.train_triples.shape[0])
    bpe = cfg.batches_per_epoch
    if bpe is None and n_train > 100_000:
        bpe = 2048
        log(f"[buddy] auto batches_per_epoch=2048 (train_q={n_train} > 100k)")
    use_capped_epoch = bpe is not None and bpe > 0 and n_train > int(bpe) * int(cfg.batch_size)
    max_batches = int(bpe) if use_capped_epoch else None
    monitor_split = "test" if state.preferred_eval_split == "test" else "val"

    log(
        f"[buddy] train {graph.bundle.name}: N={graph.num_nodes} "
        f"train_q={n_train} feats={state.use_feature} emb={model.use_embedding} "
        f"sign_k={cfg.sign_k} hops={cfg.max_hash_hops} monitor={monitor_split} "
        f"num_neg={cfg.num_negative} equal_pos_neg={int(cfg.equal_pos_neg_weight)} "
        f"same_source_neg=1"
        + (f" batches_per_epoch={max_batches} (capped)" if use_capped_epoch else " batches_per_epoch=full")
        + (f" ckpt={ckpt_path}" if ckpt_path is not None else "")
        + (f" resume_from_epoch={start_epoch}" if resumed else "")
    )

    if start_epoch > cfg.max_epochs:
        log(f"[buddy] checkpoint already finished (past max_epochs={cfg.max_epochs}); skipping train loop")
    elif epochs_no_improve >= cfg.patience and best_epoch > 0:
        log(
            f"[buddy] checkpoint already early-stopped "
            f"(epochs_no_improve={epochs_no_improve} >= patience={cfg.patience}); skipping train loop"
        )
        start_epoch = cfg.max_epochs + 1

    def _eval(split: str, max_queries: Optional[int]) -> Dict[str, float]:
        return evaluate_on_split(
            model,
            state,
            split=split,
            max_queries=max_queries,
            seed=cfg.seed,
            recall_k=cfg.recall_k,
            device=device,
            embedding=embedding,
            pair_batch_size=cfg.eval_pair_batch_size,
        )

    monitor_queries = (
        cfg.max_eval_queries if cfg.max_eval_queries is not None else cfg.max_monitor_queries
    )
    final_queries = cfg.max_eval_queries

    for epoch in range(start_epoch, cfg.max_epochs + 1):
        model.train()
        if embedding is not None:
            embedding.train()
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
            src = pos[:, 0]
            dst_pos = pos[:, 2]
            dst_neg = sample_negative_dst(graph.num_nodes, src, cfg.num_negative, generator=gen)
            src_all = src.unsqueeze(1).expand(-1, 1 + cfg.num_negative).reshape(-1)
            dst_all = torch.cat([dst_pos.unsqueeze(1), dst_neg], dim=1).reshape(-1)

            emb_table = None
            if embedding is not None:
                if model.propagate_embeddings:
                    emb_table = model.propagate_embeddings_func(embedding, graph.edge_index)
                else:
                    emb_table = embedding.weight

            logits = score_pairs(
                model,
                state.hasher,
                state.hash_tables,
                state.cards,
                src_all,
                dst_all,
                x=state.x,
                degrees=state.degrees,
                emb=emb_table,
            )
            logits_2d = logits.view(-1, 1 + cfg.num_negative)
            loss = pairwise_bce_loss(
                logits_2d, equal_pos_neg_weight=cfg.equal_pos_neg_weight
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
            val_metrics = _eval(monitor_split, monitor_queries)
            row.update({f"val/{k}": v for k, v in val_metrics.items()})
            cur = float(val_metrics.get(metric_key, 0.0))
            log(
                f"[buddy] epoch {epoch}/{cfg.max_epochs} loss={avg_loss:.4f} "
                + " ".join(f"{monitor_split}_{k}={v:.4f}" for k, v in val_metrics.items())
            )
            if cur > best_metric + 1e-6:
                best_metric = cur
                best_epoch = epoch
                epochs_no_improve = 0
                best_state_cpu, best_emb_cpu = _snapshot(model, embedding)
                log(f"[buddy] new best {metric_key}={best_metric:.4f} @ epoch {best_epoch}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= cfg.patience:
                    stop_early = True
                    log(
                        f"[buddy] early stop at epoch {epoch} "
                        f"(best {metric_key}={best_metric:.4f} @ {best_epoch})"
                    )
        else:
            log(f"[buddy] epoch {epoch}/{cfg.max_epochs} loss={avg_loss:.4f}")

        history.append(row)
        if ckpt_path is not None:
            snap, emb_snap = (
                (best_state_cpu, best_emb_cpu)
                if best_state_cpu is not None
                else _snapshot(model, embedding)
            )
            if best_state_cpu is None:
                best_state_cpu, best_emb_cpu = snap, emb_snap
                if best_epoch <= 0:
                    best_epoch = epoch
            _save_best_checkpoint(
                ckpt_path,
                state_dict=snap,
                embedding_state=emb_snap,
                cfg=cfg,
                state=state,
                best_epoch=best_epoch,
                best_metric=best_metric,
                metric_key=metric_key,
                last_completed_epoch=epoch,
                epochs_no_improve=epochs_no_improve,
                history=history,
            )
        if stop_early:
            break

    if best_state_cpu is not None:
        model.load_state_dict(best_state_cpu)
        if embedding is not None and best_emb_cpu is not None:
            embedding.load_state_dict(best_emb_cpu)
    elif ckpt_path is not None and ckpt_path.is_file():
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state_dict"])
        if embedding is not None and payload.get("embedding_state") is not None:
            embedding.load_state_dict(payload["embedding_state"])
        best_epoch = int(payload.get("best_epoch", best_epoch))
        best_metric = float(payload.get("best_val_metric", best_metric))

    test_metrics = _eval("test", final_queries)
    log(
        "[buddy] test @"
        + " ".join(f"{k}={v:.4f}" for k, v in test_metrics.items())
        + f" (best_epoch={best_epoch})"
    )
    val_metrics = _eval(monitor_split, monitor_queries)
    out = {
        "best_epoch": best_epoch,
        "best_val_metric": best_metric,
        "preferred_link_pred_eval": "recall",
        "monitor_split": monitor_split,
        "use_node_features": state.use_feature,
        "use_embedding": bool(model.use_embedding),
        "num_nodes": graph.num_nodes,
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
