"""Export raw NC graphs plus Wander protocol splits for TabPFN.

Same Wander splits as NodePFN. TabPFN inference ignores the edgelist and
applies StandardScaler + PCA (default 64).

GraphLand datasets (e.g. ``hm-categories``) are loaded with categorical
columns as **integer ordinals** (Wander / GraphPFN raw path), not one-hot.
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

from torch_geometric.data import Data

_WANDER_ROOT = Path(__file__).resolve().parents[2]
if str(_WANDER_ROOT) not in sys.path:
    sys.path.insert(0, str(_WANDER_ROOT))

from baselines.export_nc import add_nofeat_flags  # noqa: E402
from baselines.graphpfn.split_export import write_protocol_splits  # noqa: E402
from baselines.nodepfn.prepare import write_raw_tensors  # noqa: E402
from baselines.paths import WANDER_ROOT  # noqa: E402
from baselines.registry import (  # noqa: E402
    GraphPFNDatasetSpec,
    nc_eval_specs,
    resolve_nc_eval_spec,
)
from data.datasets import DATASET_REGISTRY  # noqa: E402


def resolve_prepare_root() -> Path:
    """Always write under ``raw_data/tabpfn_nc`` (or ``TABPFN_DATA_ROOT``)."""
    import os

    env = os.environ.get("TABPFN_DATA_ROOT", "").strip()
    if env:
        return Path(env)
    return WANDER_ROOT / "raw_data" / "tabpfn_nc"


def load_raw_graph(registry_key: str, data_dir: Path, *, spec: GraphPFNDatasetSpec) -> Data:
    """Load raw tensors; GraphLand uses ordinal categoricals (not one-hot)."""
    module = DATASET_REGISTRY.get(registry_key)
    if module is None:
        raise KeyError(f"Unknown registry key: {registry_key}")
    loader = getattr(module, "load_base_dataset", None)
    if loader is None:
        raise ValueError(f"{registry_key} has no load_base_dataset")

    kwargs: dict = {}
    params = inspect.signature(loader).parameters
    if spec.is_graphland:
        # Match Wander eval: raw GraphLand features, cats as integer ordinals.
        if "graphland_categorical_as_ordinals" in params:
            kwargs["graphland_categorical_as_ordinals"] = True
        if "graphland_different_transform" in params:
            kwargs["graphland_different_transform"] = True
        if not kwargs:
            # Thin wrappers: ``return load_graphland_rl(data_dir, NAME)``.
            from data.datasets._graphland_loader import load_graphland_rl

            name = getattr(module, "NAME", None)
            src = inspect.getsource(loader)
            if name and "load_graphland_rl" in src and "subsample" not in src:
                return load_graphland_rl(
                    str(data_dir),
                    name,
                    graphland_different_transform=True,
                    graphland_categorical_as_ordinals=True,
                )
    return loader(str(data_dir), **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export raw graphs + Wander splits for TabPFN NC "
            "(GraphLand categoricals as ordinals)."
        )
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
    npy_root = resolve_prepare_root()

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

    print(
        f"TabPFN prepare (raw tensors + Wander splits; "
        f"GraphLand cats=ordinals): {len(specs)} -> {npy_root}"
    )
    for spec in specs:
        data = load_raw_graph(spec.registry_key, data_dir, spec=spec)
        out_dir = npy_root / spec.slug
        info = write_raw_tensors(data, out_dir, ignore_features=spec.ignore_features)
        write_protocol_splits(
            spec, data_dir=data_dir, out_dir=out_dir, seed=args.seed
        )
        cat_note = " ordinals" if spec.is_graphland and not spec.ignore_features else ""
        print(
            f"  {spec.slug}: n={info['n_nodes']} e={info['n_edges']} "
            f"C={info['n_classes']} F={info['feat_dim']}{cat_note}"
        )
    print("Done.")


if __name__ == "__main__":
    main()
