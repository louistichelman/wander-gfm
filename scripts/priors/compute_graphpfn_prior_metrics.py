"""Sample graphs from GraphPFN's prior and compute metrics comparable to SCMPrior.

Usage (from the repo root, GraphPFN env with DGL):
    python scripts/priors/compute_graphpfn_prior_metrics.py
    python scripts/priors/compute_graphpfn_prior_metrics.py --n_graphs 50
"""

from __future__ import annotations

import argparse
import gc
import sys
import traceback
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
WANDER_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(WANDER_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from baselines.paths import GRAPHPFN_ROOT, require_checkout  # noqa: E402

require_checkout("graphpfn")

from compute_graph_metrics import (  # noqa: E402
    _build_synthetic_profile,
    compute_all_metrics,
    save_rows_csv,
)
from data.prior.native_backends import GraphPFNNativeSampler  # noqa: E402
from data.prior_to_pyg import prior_to_pyg_data  # noqa: E402


def run_graphpfn_prior_metrics(
    *,
    n_graphs: int,
    output_dir: Path,
    config_path: Path,
    pca_target_dim: int,
    row_wise_norming: bool,
    pca_before_normalization: bool,
    drop_constant_train_features: bool,
    train_device: torch.device,
    prior_device: str,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    n_workers: int,
    skip_mlp: bool = False,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_task_dir = output_dir / "by_task"
    by_task_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    sampler = GraphPFNNativeSampler(
        config_path=config_path,
        device=prior_device,
        classification_only=True,
        seed=seed,
        n_workers=n_workers,
    )

    rows: list[dict] = []
    n_errors = 0
    graph_idx = 0
    max_resample_attempts = 20

    while graph_idx < n_graphs:
        try:
            graph, X, y, d, _seq_len, train_size = sampler.sample()
            data = prior_to_pyg_data(
                graph,
                X,
                y,
                int(d.item()) if hasattr(d, "item") else int(d),
                train_size,
                pca_target_dim=pca_target_dim,
                row_wise_norming=row_wise_norming,
                pca_before_normalization=pca_before_normalization,
                drop_constant_train_features=drop_constant_train_features,
            )
            profile = _build_synthetic_profile("node_cls", data)
            row = {
                "complexity": float("nan"),
                "synthetic_task": "node_cls",
                "prior_source": "graphpfn",
                "graph_idx": graph_idx,
            }
            row.update(
                compute_all_metrics(
                    data,
                    profile,
                    train_device=train_device,
                    hidden_dim=hidden_dim,
                    epochs=epochs,
                    lr=lr,
                    weight_decay=weight_decay,
                    seed=seed,
                    complexity=None,
                    skip_mlp=skip_mlp,
                )
            )
            rows.append(row)
            graph_idx += 1

            if graph_idx % 10 == 0:
                print(f"  processed {graph_idx}/{n_graphs} (errors={n_errors})")
        except Exception:
            n_errors += 1
            traceback.print_exc()
            if n_errors >= max_resample_attempts * n_graphs:
                raise RuntimeError(
                    f"Too many sampling errors ({n_errors}); aborting."
                ) from None
        finally:
            gc.collect()
            if train_device.type == "cuda":
                torch.cuda.empty_cache()

    all_csv = output_dir / "all_graph_metrics.csv"
    task_csv = by_task_dir / "node_cls.csv"
    save_rows_csv(rows, all_csv)
    save_rows_csv(rows, task_csv)
    print(f"Saved {len(rows)} rows to {all_csv}")
    print(f"Saved {len(rows)} rows to {task_csv}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample GraphPFN prior graphs and compute graph metrics"
    )
    default_output = (
        WANDER_ROOT
        / "notebooks/results_graph_metrics/graph_metrics_graphpfn_pca9999"
    )
    default_config = GRAPHPFN_ROOT / "exp/graphpfn/pretrain/main/pretrain.toml"

    parser.add_argument("--output_dir", type=str, default=str(default_output))
    parser.add_argument("--n_graphs", type=int, default=300)
    parser.add_argument("--graphpfn_config", type=str, default=str(default_config))
    parser.add_argument("--pca_target_dim", type=int, default=9999)
    parser.add_argument(
        "--row_wise_norming",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--pca_before_normalization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When True (default): smoothing -> PCA -> normalization. "
            "When False: smoothing -> normalization -> PCA."
        ),
    )
    parser.add_argument(
        "--drop_constant_train_features",
        action="store_true",
        default=False,
        help="Drop feature columns constant on training nodes (final pipeline step).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--prior_device", type=str, default="cpu")
    parser.add_argument(
        "--train_device",
        type=str,
        default=None,
        help="Device for MLP metric training (default: cuda if available else cpu).",
    )
    parser.add_argument(
        "--skip_mlp",
        action="store_true",
        help="Skip the MLP probe (mlp_train_acc / mlp_test_acc stay NaN).",
    )
    args = parser.parse_args()

    if args.train_device is None:
        train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        train_device = torch.device(args.train_device)

    config_path = Path(args.graphpfn_config)
    if not config_path.is_file():
        raise SystemExit(f"GraphPFN config not found: {config_path}")

    output_dir = Path(args.output_dir)

    print(f"GraphPFN config: {config_path}")
    print(f"Output dir: {output_dir}")
    print(f"N graphs: {args.n_graphs}")
    print(f"PCA target dim: {args.pca_target_dim}")
    print(f"Train device: {train_device}")
    print(f"Prior device: {args.prior_device}")
    print(f"Prior workers: {args.n_workers}")

    run_graphpfn_prior_metrics(
        n_graphs=args.n_graphs,
        output_dir=output_dir,
        config_path=config_path,
        pca_target_dim=args.pca_target_dim,
        row_wise_norming=args.row_wise_norming,
        pca_before_normalization=args.pca_before_normalization,
        drop_constant_train_features=args.drop_constant_train_features,
        train_device=train_device,
        prior_device=args.prior_device,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        n_workers=args.n_workers,
        skip_mlp=args.skip_mlp,
    )


if __name__ == "__main__":
    main()
