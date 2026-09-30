"""Collect Grids per-node p(true class) maps for walk-length / walk-filter cells.

Paper cached-KV eval: proximity batches of 128 (packed-ball / new seeds),
adaptive walks off, 16 ensembles. Default walk_num is the size-tier value for
Grids (n=3125 -> 64). Phase B walks that never visit a train node can be kept
with probability ``--keep_train_free_p`` after sampling twice as many walks.

Usage (from repo root):
  python scripts/eval/collect_grids_p_true_maps.py \\
      --init_checkpoint checkpoints/pretrained_wander/pretrained_wander.pt \\
      --output_dir results/eval/grids_p_true_maps \\
      --walk_num 64 --walk_len 16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from config.arguments import (  # noqa: E402
    build_wander_config,
    get_argument_parser,
    validate_args,
)
from experiment.experiment import Experiment  # noqa: E402
from experiment.query_loader import (  # noqa: E402
    _build_bfs_proximity_batches_new_seeds,
)
from experiment.utils import set_seed  # noqa: E402
from models import Wander  # noqa: E402
from models.wander.rw_checkpoint_keys import filter_obsolete_checkpoint_keys  # noqa: E402
from models.wander.wander import _nc_combined_walk_start_count  # noqa: E402


def _parse_float_list(raw: str, *, allow_empty: bool = False) -> list[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if any(v < 0.0 or v > 1.0 for v in vals):
        raise ValueError(f"expected probabilities in [0, 1], got {raw!r}")
    if not vals and not allow_empty:
        raise ValueError(f"expected probabilities, got {raw!r}")
    return vals


def _parse_int_list(raw: str, *, allow_empty: bool = False) -> list[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if any(v <= 0 for v in vals):
        raise ValueError(f"expected positive integers, got {raw!r}")
    if not vals and not allow_empty:
        raise ValueError(f"expected positive integers, got {raw!r}")
    return vals


def _parse_seed_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _make_proximity_batches(
    edge_index: Tensor,
    num_nodes: int,
    query_ids: np.ndarray,
    batch_size: int,
    *,
    max_radius: Optional[int],
    batch_seed: Optional[int],
) -> list[Tensor]:
    query_t = torch.as_tensor(query_ids, dtype=torch.long)
    return _build_bfs_proximity_batches_new_seeds(
        edge_index, num_nodes, query_t, batch_size,
        max_radius=max_radius, batch_seed=batch_seed,
    )


def _paper_eval_argv(
    *,
    ckpt: str,
    data_dir: str,
    save_load_path: str,
    run_name: str,
    walk_num: int,
    walk_len: int,
    max_walk_len: int,
    refinements: int,
    train_kv_passes: int,
    train_kv_chunk_size: int,
    test_samples: int,
    eval_batch_size: int,
    nc_proximity_batch_max_radius: Optional[int] = None,
) -> list[str]:
    argv = [
        "--init_checkpoint", ckpt,
        "--eval_only",
        "--data_dir", data_dir,
        "--save_load_path", save_load_path,
        "--run_name", run_name,
        "--test_datasets", "GRIDS",
        "--ignore_features",
        "--batch_size_link", "4",
        "--eval_batch_size_link", "8",
        "--batch_size_node", "16",
        "--eval_batch_size_node", str(eval_batch_size),
        "--wander_walk_num", str(walk_num),
        "--wander_eval_walk_num", str(walk_num),
        "--wander_walk_len", str(walk_len),
        "--wander_max_walk_len", str(max_walk_len),
        "--wander_train_kv_walk_num", str(walk_num),
        "--wander_global_attention_update",
        "--wander_inter_node_chunksize", "128",
        "--wander_intra_node_chunksize", "2048",
        "--wander_refinements", str(refinements),
        "--wander_checkpoint_refinements",
        "--wander_rotary_emb_intra_node_attention",
        "--wander_feature_embedding_mlp",
        "--wander_randomize_feat_columns",
        "--wander_randomize_label_columns",
        "--wander_net", "gru",
        "--node_cls_random_training_batches",
        "--add_inverse_edges_KGs",
        "--num_negatives", "512",
        "--wander_test_samples", str(test_samples),
        "--wander_cached_train_kv",
        "--nc_proximity_batching",
    ]
    if nc_proximity_batch_max_radius is not None:
        argv.extend(
            ["--nc_proximity_batch_max_radius", str(nc_proximity_batch_max_radius)]
        )
    argv.extend([
        "--wander_train_kv_passes", str(train_kv_passes),
        "--wander_train_kv_chunk_size", str(train_kv_chunk_size),
        "--pca_target_dim", "64",
        "--pca_target_dim_link", "32",
        "--pca_target_dim_link_syn", "32",
        "--pca_target_dim_node_syn", "64",
        "--no-pca_before_normalization",
        "--graphland_categorical_as_ordinals",
        "--graphland_different_transform",
        "--drop_constant_train_features",
        "--final_inductive_zscore",
        "--no_wander_adaptive_walks",
        "--seeds", "0",
    ])
    return argv


def _load_state_dict(path: str, device: torch.device) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def _set_walks(model: Wander, walk_num: int, walk_len: int, *, phase_a_walk_num: int) -> None:
    if walk_len > int(model.L_max):
        raise ValueError(f"walk_len={walk_len} exceeds max_walk_len={model.L_max}")
    model.N = int(walk_num)
    model.eval_walk_num = int(walk_num)
    model.train_kv_walk_num = int(phase_a_walk_num)
    model.L = int(walk_len)


def _true_class_vec(data: Data) -> np.ndarray:
    y = data.y
    if y.ndim == 1:
        return y.detach().cpu().numpy().astype(np.int64)
    return y.argmax(dim=-1).detach().cpu().numpy().astype(np.int64)


def _grid_layout(data: Data, n_nodes: int) -> tuple[np.ndarray, int, int, int]:
    grid_id_attr = getattr(data, "grid_id", None)
    if grid_id_attr is not None:
        grid_id = grid_id_attr.view(-1).detach().cpu().numpy().astype(np.int64)
    else:
        n_per = 25 * 25
        grid_id = (np.arange(n_nodes) // n_per).astype(np.int64)
    h = int(getattr(data, "grid_h", 25))
    w = int(getattr(data, "grid_w", 25))
    n_grids = int(getattr(data, "n_grids", int(grid_id.max()) + 1 if grid_id.size else 1))
    return grid_id, h, w, n_grids


def _choose_viz_grid(
    grid_id: np.ndarray,
    train_mask: np.ndarray,
    *,
    h: int,
    w: int,
    requested: Optional[int],
) -> int:
    n_grids = int(grid_id.max()) + 1 if grid_id.size else 1
    if requested is not None:
        if requested < 0 or requested >= n_grids:
            raise ValueError(f"--viz_grid {requested} out of range [0, {n_grids})")
        return int(requested)
    n_per = h * w
    best_gid = 0
    best_sep = -1
    for gid in range(n_grids):
        train = np.flatnonzero(train_mask & (grid_id == gid))
        if train.size < 2:
            sep = 0
        else:
            locals_ = train % n_per
            rs, cs = locals_ // w, locals_ % w
            sep = int(np.abs(rs[:, None] - rs[None, :]).max() + np.abs(cs[:, None] - cs[None, :]).max())
        if sep > best_sep:
            best_sep = sep
            best_gid = gid
    return int(best_gid)


def _filter_train_free_walks(
    walk_tuple: tuple,
    train_mask: Tensor,
    keep_prob: float,
) -> tuple[tuple, dict[str, float]]:
    """Keep train-hitting walks always; keep train-free walks with probability p."""
    walks = walk_tuple[0]
    if walks.ndim != 4:
        raise ValueError(f"expected walks [T, B, N, L], got {tuple(walks.shape)}")
    n_layers, n_graphs, n_walks, _walk_len = walks.shape
    if n_graphs != 1:
        raise ValueError(f"expected B=1 walk tensor, got B={n_graphs}")

    tm = train_mask.to(device=walks.device, dtype=torch.bool)
    valid = (walks >= 0) & (walks < tm.numel())
    safe = walks.clamp(min=0, max=max(int(tm.numel()) - 1, 0))
    hits = (tm[safe] & valid).any(dim=-1)  # [T, 1, N]
    if keep_prob >= 1.0:
        stats = {
            "n_walks_sampled": float(n_layers * n_walks),
            "n_walks_hit_train": float(hits.sum().item()),
            "n_walks_kept": float(n_layers * n_walks),
            "n_walks_train_free_kept": float((~hits).sum().item()),
            "n_walks_padded": 0.0,
        }
        return walk_tuple, stats

    keep = hits.clone()
    if keep_prob > 0.0:
        rand = torch.rand(hits.shape, device=walks.device)
        keep = hits | ((~hits) & (rand < keep_prob))

    keep_idx: list[Tensor] = []
    n_kept_layers: list[int] = []
    for t in range(n_layers):
        idx = keep[t, 0].nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            idx = torch.zeros(1, dtype=torch.long, device=walks.device)
        keep_idx.append(idx)
        n_kept_layers.append(int(idx.numel()))
    max_n = max(n_kept_layers)
    n_padded = sum(max_n - n for n in n_kept_layers)

    filtered: list[Tensor] = []
    for ch in walk_tuple:
        parts = []
        for t, idx in enumerate(keep_idx):
            sl = ch[t, 0].index_select(0, idx)
            if sl.shape[0] < max_n:
                pad_n = max_n - sl.shape[0]
                sl = torch.cat([sl, sl[-1:].expand(pad_n, *sl.shape[1:])], dim=0)
            parts.append(sl)
        filtered.append(torch.stack(parts, dim=0).unsqueeze(1))

    n_hit = float(hits.sum().item())
    n_kept = float(sum(n_kept_layers))
    stats = {
        "n_walks_sampled": float(n_layers * n_walks),
        "n_walks_hit_train": n_hit,
        "n_walks_kept": n_kept,
        "n_walks_train_free_kept": float(n_kept - n_hit),
        "n_walks_padded": float(n_padded),
    }
    return tuple(filtered), stats


def _install_train_free_filter(
    model: Wander,
    train_mask: Tensor,
    keep_prob: float,
    stats_out: list[dict[str, float]],
) -> Any:
    orig = model.walks_node_classification

    def wrapped(
        data,
        batch_indices,
        stay_on_cpu: bool = False,
        start_mode: str = "default",
        walk_focus_indices=None,
    ):
        walk_tuple = orig(
            data,
            batch_indices,
            stay_on_cpu=stay_on_cpu,
            start_mode=start_mode,
            walk_focus_indices=walk_focus_indices,
        )
        if start_mode != "batch_only":
            return walk_tuple
        filtered, draw_stats = _filter_train_free_walks(walk_tuple, train_mask, keep_prob)
        stats_out.append(draw_stats)
        return filtered

    model.walks_node_classification = wrapped
    return orig


def _mean_stats(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    out = {k: float(sum(r[k] for r in rows) / len(rows)) for k in keys}
    out["n_draws"] = float(len(rows))
    return out


def _cell_name(walk_num: int, walk_len: int, keep_p: Optional[float]) -> str:
    if keep_p is None:
        return f"n{walk_num}_l{walk_len}"
    if float(keep_p) == int(keep_p):
        return f"n{walk_num}_l{walk_len}_p{int(keep_p)}"
    return f"n{walk_num}_l{walk_len}_p{keep_p}"


@torch.no_grad()
def _eval_cell(
    model: Wander,
    data: Data,
    batches: list[Tensor],
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
    query_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(data.num_nodes)
    n_cls = int(data.y.shape[1]) if data.y.ndim == 2 else int(data.y.max().item()) + 1
    logits_sum = torch.zeros(n, n_cls, dtype=torch.float64)
    logits_count = torch.zeros(n, dtype=torch.float64)
    use_cuda = device.type == "cuda"
    n_done = 0
    for batch in batches:
        batch = batch.to(device)
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda):
            logits = model(batch, data)
        logits = logits.detach().float().cpu()
        ids = batch.detach().cpu()
        logits_sum[ids] += logits.double()
        logits_count[ids] += 1.0
        n_done += int(ids.numel())
        print(f"    scored {n_done}/{len(query_ids)} nodes", flush=True)
    mean = torch.full((n, n_cls), float("nan"), dtype=torch.float64)
    ok = logits_count > 0
    mean[ok] = logits_sum[ok] / logits_count[ok].unsqueeze(-1)
    p_true = np.full(n, np.nan, dtype=np.float64)
    y = _true_class_vec(data)
    probs = torch.softmax(mean.float(), dim=-1).numpy()
    queried = ok.numpy()
    p_true[queried] = probs[queried, y[queried]]
    pred = np.full(n, -1, dtype=np.int64)
    pred[queried] = np.nanargmax(probs[queried], axis=-1)
    return p_true, pred, probs


def _hop_to_train(grid_id: np.ndarray, train_mask: np.ndarray, *, h: int, w: int) -> np.ndarray:
    n = int(grid_id.size)
    n_per = h * w
    local = np.arange(n) % n_per
    r, c = local // w, local % w
    dist = np.full(n, n_per, dtype=np.int64)
    for t in np.flatnonzero(train_mask):
        same = grid_id == int(grid_id[t])
        tr, tc = divmod(int(t) % n_per, w)
        cand = np.abs(r - tr) + np.abs(c - tc)
        dist = np.where(same, np.minimum(dist, cand), dist)
    return dist


def _plot_one_grid(
    p_true: np.ndarray,
    grid_id: np.ndarray,
    train_mask: np.ndarray,
    *,
    gid: int,
    h: int,
    w: int,
    title: str,
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError:
        print(f"matplotlib not available; skip {out_path}")
        return

    n_per = h * w
    sel = grid_id == gid
    ids = np.flatnonzero(sel)
    field = np.full((h, w), np.nan, dtype=np.float64)
    locals_ = ids % n_per
    field[locals_ // w, locals_ % w] = p_true[ids]
    train_ids = np.flatnonzero(sel & train_mask)
    train_local = train_ids % n_per
    tr, tc = train_local // w, train_local % w

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    im = ax.imshow(
        field,
        origin="upper",
        cmap="viridis",
        norm=Normalize(vmin=0.0, vmax=1.0),
        interpolation="nearest",
    )
    if train_ids.size:
        ax.scatter(
            tc, tr, marker="*", s=220, c="red", edgecolors="white", linewidths=0.8,
            zorder=3, label="labeled context",
        )
    ax.set_title(title)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.set_xticks([0, w // 2, w - 1])
    ax.set_yticks([0, h // 2, h - 1])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("p(correct class)")
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def _accuracy(pred: np.ndarray, y: np.ndarray, mask: np.ndarray) -> Optional[float]:
    sel = mask & (pred >= 0)
    if not np.any(sel):
        return None
    return float((pred[sel] == y[sel]).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init_checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="raw_data")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--test_samples", type=int, default=16)
    parser.add_argument("--refinements", type=int, default=6)
    parser.add_argument("--max_walk_len", type=int, default=128)
    parser.add_argument("--train_kv_passes", type=int, default=16)
    parser.add_argument("--train_kv_chunk_size", type=int, default=20000)
    parser.add_argument("--walk_num", type=int, default=64, help="Default size-tier walk_num for Grids.")
    parser.add_argument("--walk_len", type=int, default=None)
    parser.add_argument("--walk_lens", type=str, default="", help="Comma list; overrides --walk_len.")
    parser.add_argument(
        "--keep_train_free_p",
        type=float,
        default=None,
        help="Keep train-free Phase B walks with this probability. None = no filter.",
    )
    parser.add_argument(
        "--keep_train_free_ps",
        type=str,
        default="",
        help="Comma list of p values; implies 2x --walk_num Phase B walks.",
    )
    parser.add_argument(
        "--filter_walk_num_mult",
        type=int,
        default=2,
        help="Phase B walk_num multiplier when filtering train-free walks.",
    )
    parser.add_argument("--filter_walk_len", type=int, default=64)
    parser.add_argument("--viz_grid", type=int, default=None)
    parser.add_argument(
        "--nc_proximity_batch_max_radius",
        type=int,
        default=None,
        help="Max hop radius for packed-ball batches. Unset grows until batch_size / component.",
    )
    parser.add_argument(
        "--proximity_batch_seeds",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help=(
            "Comma list of new-seed batch-partition RNG seeds. Maps are the mean "
            "softmax over those partitions (blurs batch borders). Empty = one "
            "deterministic partition."
        ),
    )
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.eval_batch_size < 1:
        raise ValueError("--eval_batch_size must be >= 1")
    if args.test_samples < 1:
        raise ValueError("--test_samples must be >= 1")
    if args.walk_num < 1:
        raise ValueError("--walk_num must be >= 1")
    if args.filter_walk_num_mult < 1:
        raise ValueError("--filter_walk_num_mult must be >= 1")

    lens = _parse_int_list(args.walk_lens, allow_empty=True)
    if not lens and args.walk_len is not None:
        lens = [int(args.walk_len)]
    keep_ps = _parse_float_list(args.keep_train_free_ps, allow_empty=True)
    if not keep_ps and args.keep_train_free_p is not None:
        keep_ps = [float(args.keep_train_free_p)]
    if not lens and not keep_ps:
        raise ValueError("need --walk_len/--walk_lens and/or --keep_train_free_p(s)")
    if lens and max(lens) > args.max_walk_len:
        raise ValueError("all walk lengths must be <= --max_walk_len")
    if args.filter_walk_len > args.max_walk_len:
        raise ValueError("--filter_walk_len must be <= --max_walk_len")
    if args.nc_proximity_batch_max_radius is not None:
        if int(args.nc_proximity_batch_max_radius) < 0:
            raise ValueError("--nc_proximity_batch_max_radius must be >= 0")

    batch_seeds: list[Optional[int]]
    parsed_seeds = _parse_seed_list(args.proximity_batch_seeds)
    batch_seeds = parsed_seeds if parsed_seeds else [None]

    cells: list[tuple[int, int, Optional[float], int]] = []
    for walk_len in lens:
        cells.append((args.walk_num, walk_len, None, args.walk_num))
    for p in keep_ps:
        cells.append(
            (
                args.walk_num * args.filter_walk_num_mult,
                args.filter_walk_len,
                float(p),
                args.walk_num,
            )
        )

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt = str(Path(args.init_checkpoint).resolve())
    if not Path(ckpt).is_file():
        raise FileNotFoundError(ckpt)

    first_n, first_l, _, first_a = cells[0]
    wander_argv = _paper_eval_argv(
        ckpt=ckpt,
        data_dir=args.data_dir,
        save_load_path=str(out_root),
        run_name="grids_p_true_maps",
        walk_num=first_n,
        walk_len=first_l,
        max_walk_len=args.max_walk_len,
        refinements=args.refinements,
        train_kv_passes=args.train_kv_passes,
        train_kv_chunk_size=args.train_kv_chunk_size,
        test_samples=args.test_samples,
        eval_batch_size=args.eval_batch_size,
        nc_proximity_batch_max_radius=args.nc_proximity_batch_max_radius,
    )
    wander_parser = get_argument_parser()
    wander_args = wander_parser.parse_args(wander_argv)
    validate_args(wander_args)

    experiment = Experiment(wander_args, rank=0, world_size=1)
    device = torch.device(args.device if args.device is not None else experiment.device)
    amp_dtype = experiment.amp_dtype

    set_seed(args.seed)
    _, datasets_test = experiment.load_datasets(args.seed)
    bundle = datasets_test.get("Grids") or datasets_test.get("GRIDS")
    if bundle is None:
        raise RuntimeError(f"GRIDS not in loaded test datasets: {list(datasets_test)}")
    from data.graph_bundle import resolve_split
    data = resolve_split(bundle, "test").to(device)
    if getattr(data, "num_relations", None) is None:
        data.num_relations = 1
    if getattr(data, "edge_type", None) is None:
        data.edge_type = torch.zeros(int(data.edge_index.size(1)), dtype=torch.long, device=device)

    n_nodes = int(data.num_nodes)
    train_mask = data.train_mask.detach().cpu().numpy().astype(bool)
    val_mask = data.val_mask.detach().cpu().numpy().astype(bool)
    test_mask = data.test_mask.detach().cpu().numpy().astype(bool)
    y = _true_class_vec(data)
    grid_id, grid_h, grid_w, n_grids = _grid_layout(data, n_nodes)
    hops = _hop_to_train(grid_id, train_mask, h=grid_h, w=grid_w)
    viz_grid = _choose_viz_grid(
        grid_id, train_mask, h=grid_h, w=grid_w, requested=args.viz_grid,
    )
    query_ids = np.flatnonzero(~train_mask).astype(np.int64)
    n_cls = int(data.y.shape[1]) if data.y.ndim == 2 else int(data.y.max().item()) + 1
    preview_batches = _make_proximity_batches(
        data.edge_index.cpu(), n_nodes, query_ids, args.eval_batch_size,
        max_radius=args.nc_proximity_batch_max_radius,
        batch_seed=batch_seeds[0],
    )
    batch_mode = (
        "packed_ball"
        if args.nc_proximity_batch_max_radius is None
        else f"packed_ball_r{args.nc_proximity_batch_max_radius}"
    )
    if batch_seeds[0] is not None:
        batch_mode += f"_avg{len(batch_seeds)}"

    model_cfg = build_wander_config(wander_args)
    model = Wander(model_cfg).to(device)
    state = filter_obsolete_checkpoint_keys(_load_state_dict(ckpt, device))
    model.load_state_dict(state, strict=False)
    model.eval()
    model.use_ensemble_prefetch = False
    model._ensemble_eval_mode = "sequential"

    print(
        f"Grids n={n_nodes} train={int(train_mask.sum())} val={int(val_mask.sum())} "
        f"test={int(test_mask.sum())} query={query_ids.size} "
        f"batches~{len(preview_batches)} batch_mode={batch_mode} "
        f"batch_seeds={batch_seeds} viz_grid={viz_grid} cells={len(cells)}",
        flush=True,
    )

    summary_rows: list[dict[str, Any]] = []
    train_mask_t = data.train_mask.detach()

    for walk_num, walk_len, keep_p, phase_a_n in cells:
        name = _cell_name(walk_num, walk_len, keep_p)
        cell_dir = out_root / name
        maps_path = cell_dir / "maps.npz"
        if args.skip_existing and maps_path.is_file():
            print(f"=== skip {name} (exists) ===", flush=True)
            continue
        cell_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"=== {name}: Phase A N={phase_a_n} L={walk_len}  "
            f"Phase B N={walk_num} L={walk_len} keep_p={keep_p} ===",
            flush=True,
        )
        _set_walks(model, walk_num, walk_len, phase_a_walk_num=phase_a_n)
        n_starts = _nc_combined_walk_start_count(walk_num, int(model.T))
        walks_per_layer = max(1, n_starts // int(model.T))
        print(f"  Phase B walks/layer (before filter)={walks_per_layer}", flush=True)

        set_seed(args.seed + 101)
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            cache = model.precompute_train_kv_cache(data, num_passes=args.train_kv_passes)
        model.attach_train_kv_cache(cache)

        filter_stats: list[dict[str, float]] = []
        orig_walks = None
        if keep_p is not None:
            orig_walks = _install_train_free_filter(model, train_mask_t, float(keep_p), filter_stats)

        probs_sum = np.zeros((n_nodes, n_cls), dtype=np.float64)
        probs_n = np.zeros(n_nodes, dtype=np.float64)
        for bseed in batch_seeds:
            batches = _make_proximity_batches(
                data.edge_index.cpu(), n_nodes, query_ids, args.eval_batch_size,
                max_radius=args.nc_proximity_batch_max_radius,
                batch_seed=bseed,
            )
            set_seed(args.seed + 17 * (0 if bseed is None else int(bseed) + 1))
            print(
                f"  batch_seed={bseed} n_batches={len(batches)}",
                flush=True,
            )
            _, _, probs = _eval_cell(
                model, data, batches, device=device, amp_dtype=amp_dtype, query_ids=query_ids,
            )
            ok = np.isfinite(probs).all(axis=1)
            probs_sum[ok] += probs[ok]
            probs_n[ok] += 1.0
        if orig_walks is not None:
            model.walks_node_classification = orig_walks
        model.attach_train_kv_cache(None)
        cache.release()

        mean_probs = np.full((n_nodes, n_cls), np.nan, dtype=np.float64)
        scored = probs_n > 0
        mean_probs[scored] = probs_sum[scored] / probs_n[scored, None]
        p_true = np.full(n_nodes, np.nan, dtype=np.float64)
        p_true[scored] = mean_probs[scored, y[scored]]
        pred = np.full(n_nodes, -1, dtype=np.int64)
        pred[scored] = np.nanargmax(mean_probs[scored], axis=-1)

        test_acc = _accuracy(pred, y, test_mask)
        val_acc = _accuracy(pred, y, val_mask)
        hop_means: dict[str, float] = {}
        for hop in sorted(set(int(h) for h in hops[~train_mask])):
            sel = (~train_mask) & (hops == hop) & np.isfinite(p_true)
            if np.any(sel):
                hop_means[str(hop)] = float(np.nanmean(p_true[sel]))

        filt_mean = _mean_stats(filter_stats)
        meta = {
            "run_name": name,
            "walk_num": walk_num,
            "walk_len": walk_len,
            "phase_a_walk_num": phase_a_n,
            "phase_a_walk_len": walk_len,
            "keep_train_free_p": keep_p,
            "eval_batch_size": args.eval_batch_size,
            "nc_proximity_batching": True,
            "nc_proximity_batch_max_radius": args.nc_proximity_batch_max_radius,
            "proximity_batch_mode": batch_mode,
            "proximity_batch_seeds": [
                s if s is not None else None for s in batch_seeds
            ],
            "test_samples": args.test_samples,
            "walks_per_layer_before_filter": walks_per_layer,
            "test_accuracy": test_acc,
            "val_accuracy": val_acc,
            "mean_p_true_query": float(np.nanmean(p_true[query_ids])),
            "viz_grid": viz_grid,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "n_grids": n_grids,
            "n_query": int(query_ids.size),
            "mean_p_true_by_hop": hop_means,
            "filter_stats": filt_mean,
            "seed": args.seed,
        }
        np.savez_compressed(
            maps_path,
            p_true=p_true,
            pred=pred,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            grid_id=grid_id,
            hops=hops,
        )
        (cell_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        title = f"walk_num={walk_num}  walk_len={walk_len}"
        if keep_p is not None:
            title += f"  keep_p={keep_p}"
        title += f"  grid={viz_grid}"
        _plot_one_grid(
            p_true, grid_id, train_mask,
            gid=viz_grid, h=grid_h, w=grid_w, title=title,
            out_path=cell_dir / f"grid_{viz_grid}.png",
        )
        print(
            f"  test_acc={test_acc} val_acc={val_acc} "
            f"mean_p_true={meta['mean_p_true_query']:.4f}",
            flush=True,
        )
        summary_rows.append(meta)

    summary_path = out_root / "cells_this_job.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
