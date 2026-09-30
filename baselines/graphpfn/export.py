#!/usr/bin/env python3
"""Export multi-relational Wander graphs as homogeneous GraphPFN npy bundles.

Datasets registered with ``export_homogeneous=True`` in ``baselines.registry``
are loaded via their ``load_base_dataset`` hook, collapsed to a single edge type
(duplicates across relations removed), and written as:

* ``features.npy``, ``targets.npy``, ``edgelist.npy`` (shape ``[E, 2]``)
* Wander NC-protocol ``split.npz`` (and ``split_{i}.npz`` when several splits)

Usage (from the repo root):

    python baselines/graphpfn/export.py
    python baselines/graphpfn/export.py --datasets CORA,CITESEER
    python baselines/graphpfn/export.py --data_dir raw_data
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, is_undirected, to_undirected

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.graphpfn.split_export import (  # noqa: E402
    sibling_featured_dir,
    write_protocol_splits,
)
from baselines.paths import WANDER_ROOT, GRAPHPFN_ROOT, require_checkout  # noqa: E402
from baselines.registry import (  # noqa: E402
    GRAPHPFN_DATASET_SPECS,
    GraphPFNDatasetSpec,
    spec_by_slug,
)
from data.datasets import DATASET_REGISTRY  # noqa: E402


def collapse_to_homogeneous(data: Data) -> torch.Tensor:
    """Merge all relation types into one undirected, deduplicated edge_index."""
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1
    if edge_index.numel() == 0:
        return edge_index
    if not is_undirected(edge_index, num_nodes=num_nodes):
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    return coalesce(edge_index, num_nodes=num_nodes)


def edges_for_export(data: Data, relation_index: int | None) -> torch.Tensor:
    """Homogeneous edge_index for export; optionally keep one relation type only."""
    if relation_index is None:
        return collapse_to_homogeneous(data)

    edge_type = getattr(data, "edge_type", None)
    if edge_type is None:
        raise ValueError(
            f"relation_index={relation_index} requires edge_type on loaded graph"
        )
    num_rels = int(getattr(data, "num_relations", 1) or 1)
    if relation_index < 0 or relation_index >= num_rels:
        raise ValueError(
            f"relation_index={relation_index} out of range for {num_rels} relations"
        )

    mask = edge_type == relation_index
    edge_index = data.edge_index[:, mask]
    num_nodes = data.num_nodes
    if num_nodes is None:
        num_nodes = int(data.edge_index.max().item()) + 1
    if edge_index.numel() == 0:
        return edge_index
    if not is_undirected(edge_index, num_nodes=num_nodes):
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    return coalesce(edge_index, num_nodes=num_nodes)


def load_export_graph(spec: GraphPFNDatasetSpec, data_dir: Path) -> Data:
    module = DATASET_REGISTRY.get(spec.registry_key)
    if module is None:
        raise KeyError(f"Unknown registry key: {spec.registry_key}")
    loader = getattr(module, "load_base_dataset", None)
    if loader is None:
        raise ValueError(
            f"{spec.registry_key} has no load_base_dataset; "
            "only node-classification multiplex datasets are supported."
        )
    return loader(str(data_dir))


def _export_nofeat_from_featured_npy(
    spec: GraphPFNDatasetSpec,
    *,
    data_dir: Path,
    out_dir: Path,
    seed: int,
) -> bool:
    """Reuse a featured sibling npy graph when ``load_base_dataset`` is unavailable."""
    sibling = sibling_featured_dir(spec, out_dir)
    if sibling is None:
        return False
    if not all((sibling / name).is_file() for name in ("edgelist.npy", "targets.npy")):
        return False
    targets = np.load(sibling / "targets.npy")
    n = int(targets.shape[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "features.npy", np.zeros((n, 0), dtype=np.float32))
    np.save(out_dir / "targets.npy", targets.astype(np.float32, copy=False))
    shutil.copy2(sibling / "edgelist.npy", out_dir / "edgelist.npy")
    write_protocol_splits(spec, data_dir=data_dir, out_dir=out_dir, seed=seed)
    edges = np.load(out_dir / "edgelist.npy")
    print(
        f"  exported {spec.slug}: nodes={n}, edges={int(edges.shape[0])} "
        f"(from {sibling.name} npy, feat_dim=0) -> {out_dir}"
    )
    return True


def export_multiplex_dataset(
    spec: GraphPFNDatasetSpec,
    *,
    data_dir: Path,
    out_dir: Path,
    seed: int = 42,
) -> None:
    """Write homogeneous npy bundle for one multiplex dataset."""
    if spec.ignore_features and _export_nofeat_from_featured_npy(
        spec, data_dir=data_dir, out_dir=out_dir, seed=seed
    ):
        return
    data = load_export_graph(spec, data_dir)
    edge_index = edges_for_export(data, spec.relation_index)

    out_dir.mkdir(parents=True, exist_ok=True)
    if spec.ignore_features:
        feat_arr = np.zeros((data.num_nodes, 0), dtype=np.float32)
        feat_dim = 0
    else:
        feat_arr = data.x.cpu().numpy().astype(np.float32)
        feat_dim = data.x.size(1)
    np.save(out_dir / "features.npy", feat_arr)
    np.save(out_dir / "targets.npy", data.y.cpu().numpy().astype(np.float32))
    np.save(out_dir / "edgelist.npy", edge_index.cpu().numpy().T.astype(np.int64))
    write_protocol_splits(spec, data_dir=data_dir, out_dir=out_dir, seed=seed)

    num_rels = int(getattr(data, "num_relations", 1) or 1)
    rel_note = (
        f"rel={spec.relation_index}/{num_rels}"
        if spec.relation_index is not None
        else f"{num_rels} rels collapsed"
    )
    print(
        f"  exported {spec.slug}: nodes={data.num_nodes}, "
        f"edges={edge_index.size(1)} (from {data.edge_index.size(1)} typed, "
        f"{rel_note}), feat_dim={feat_dim} -> {out_dir}"
    )


def export_spec(
    spec: GraphPFNDatasetSpec,
    *,
    data_dir: Path,
    out_root: Path,
    seed: int = 42,
) -> Path:
    out_dir = out_root / spec.slug
    export_multiplex_dataset(spec, data_dir=data_dir, out_dir=out_dir, seed=seed)
    return out_dir


def export_homogeneous_specs(
    *,
    registry_keys: set[str] | None = None,
) -> list[GraphPFNDatasetSpec]:
    specs = [s for s in GRAPHPFN_DATASET_SPECS if s.export_homogeneous]
    if registry_keys is None:
        return specs
    return [s for s in specs if s.registry_key in registry_keys]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export homogeneous GraphPFN npy bundles for multiplex datasets."
    )
    parser.add_argument("--data_dir", type=str, default="raw_data")
    parser.add_argument(
        "--out_root",
        type=str,
        default="",
        help="Root directory for per-slug export folders (default: third_party/graphpfn/data).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated registry keys (default: all export_homogeneous specs).",
    )
    parser.add_argument(
        "--slug",
        type=str,
        default="",
        help="Export a single dataset slug (overrides --datasets).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.slug.strip():
        specs = [spec_by_slug(args.slug.strip())]
        if not specs[0].export_homogeneous:
            raise SystemExit(f"Slug {args.slug!r} is not marked export_homogeneous.")
    elif args.datasets.strip():
        keys = {k.strip().upper() for k in args.datasets.split(",") if k.strip()}
        specs = export_homogeneous_specs(registry_keys=keys)
        missing = keys - {s.registry_key for s in specs}
        if missing:
            raise SystemExit(
                f"Unknown or non-export registry keys: {sorted(missing)}"
            )
    else:
        specs = export_homogeneous_specs()

    if not specs:
        raise SystemExit("No multiplex datasets selected for export.")

    require_checkout("graphpfn")
    out_root = Path(args.out_root) if args.out_root else GRAPHPFN_ROOT / "data"
    if not out_root.is_absolute():
        out_root = (WANDER_ROOT / out_root).resolve()
    else:
        out_root = out_root.resolve()

    print(f"Exporting {len(specs)} dataset(s) to {out_root}")
    for spec in specs:
        rel = args.data_dir if args.data_dir not in {"data", "raw_data"} else spec.export_data_dir
        data_dir = Path(rel)
        if not data_dir.is_absolute():
            # Prefer Wander raw_data; fall back to the spec's export_data_dir under WANDER_ROOT.
            cand = WANDER_ROOT / args.data_dir
            data_dir = cand.resolve() if cand.exists() else (WANDER_ROOT / spec.export_data_dir).resolve()
        print(f"{spec.registry_key} (data_dir={data_dir})")
        export_spec(spec, data_dir=data_dir, out_root=out_root, seed=args.seed)
    print("Done.")


if __name__ == "__main__":
    main()
