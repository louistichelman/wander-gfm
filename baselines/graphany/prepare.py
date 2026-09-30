"""Export raw NC graphs plus Wander protocol splits for GraphAny.

Features and edges come from ``load_base_dataset`` (no Wander z-score / PCA).
Masks come from ``write_protocol_splits`` (official multi-splits or GraphAny
20-per-class). GraphAny inference applies its own bidirect / LinearGNN
preprocess.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from torch_geometric.data import Data

_WANDER_ROOT = Path(__file__).resolve().parents[2]
if str(_WANDER_ROOT) not in sys.path:
    sys.path.insert(0, str(_WANDER_ROOT))

from baselines.export_nc import add_nofeat_flags, labels_1d  # noqa: E402
from baselines.graphpfn.split_export import write_protocol_splits  # noqa: E402
from baselines.paths import WANDER_ROOT, resolve_graphany_data_root  # noqa: E402
from baselines.registry import nc_eval_specs, resolve_nc_eval_spec  # noqa: E402
from data.datasets import DATASET_REGISTRY  # noqa: E402


def load_raw_graph(registry_key: str, data_dir: Path) -> Data:
    module = DATASET_REGISTRY.get(registry_key)
    if module is None:
        raise KeyError(f"Unknown registry key: {registry_key}")
    loader = getattr(module, "load_base_dataset", None)
    if loader is None:
        raise ValueError(f"{registry_key} has no load_base_dataset")
    return loader(str(data_dir))


def write_raw_tensors(data: Data, out_dir: Path, *, ignore_features: bool) -> dict[str, int]:
    """Write ``features.npy``, ``targets.npy``, ``edgelist.npy`` from a raw PyG graph."""
    n_nodes = int(data.num_nodes)
    if ignore_features or getattr(data, "x", None) is None:
        features = np.zeros((n_nodes, 0), dtype=np.float32)
    else:
        features = data.x.detach().cpu().numpy().astype(np.float32)
    labels = labels_1d(data.y.cpu().numpy())
    edge_index = data.edge_index
    if edge_index is None or edge_index.numel() == 0:
        edgelist = np.zeros((0, 2), dtype=np.int64)
    else:
        edgelist = edge_index.detach().cpu().numpy().T.astype(np.int64)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "features.npy", features)
    np.save(out_dir / "targets.npy", labels.astype(np.float32))
    np.save(out_dir / "edgelist.npy", edgelist)
    n_classes = int(labels[labels >= 0].max()) + 1 if np.any(labels >= 0) else 0
    return {
        "n_nodes": n_nodes,
        "n_edges": int(edgelist.shape[0]),
        "n_classes": n_classes,
        "feat_dim": int(features.shape[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export raw graphs + Wander splits for GraphAny NC."
    )
    parser.add_argument("--data_dir", type=str, default="raw_data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--datasets", type=str, default="")
    add_nofeat_flags(parser)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (WANDER_ROOT / data_dir).resolve()
    npy_root = resolve_graphany_data_root()

    if args.dataset.strip():
        specs = [resolve_nc_eval_spec(args.dataset.strip(), nofeat=args.nofeat)]
    elif args.datasets.strip():
        specs = [
            resolve_nc_eval_spec(k.strip(), nofeat=args.nofeat)
            for k in args.datasets.split(",")
            if k.strip()
        ]
    else:
        specs = list(nc_eval_specs(nofeat=True if args.nofeat else False))

    print(f"GraphAny prepare (raw tensors + Wander splits): {len(specs)} -> {npy_root}")
    for spec in specs:
        data = load_raw_graph(spec.registry_key, data_dir)
        out_dir = npy_root / spec.slug
        info = write_raw_tensors(data, out_dir, ignore_features=spec.ignore_features)
        write_protocol_splits(
            spec, data_dir=data_dir, out_dir=out_dir, seed=args.seed
        )
        print(
            f"  {spec.slug}: n={info['n_nodes']} e={info['n_edges']} "
            f"C={info['n_classes']} F={info['feat_dim']}"
        )
    print("Done.")


if __name__ == "__main__":
    main()
