"""Prepare AnyGraph NC node_data pickles from Wander datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_WANDER_ROOT = Path(__file__).resolve().parents[2]
if str(_WANDER_ROOT) not in sys.path:
    sys.path.insert(0, str(_WANDER_ROOT))

from baselines.export_nc import add_nofeat_flags, export_nc_spec  # noqa: E402
from baselines.paths import WANDER_ROOT, resolve_anygraph_nc_data_root  # noqa: E402
from baselines.registry import nc_eval_specs, resolve_nc_eval_spec  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Wander graphs for AnyGraph NC.")
    parser.add_argument("--data_dir", type=str, default="raw_data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--datasets", type=str, default="")
    add_nofeat_flags(parser)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (WANDER_ROOT / data_dir).resolve()
    anygraph_root = resolve_anygraph_nc_data_root()

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

    print(f"AnyGraph NC prepare: {len(specs)} dataset(s) -> {anygraph_root}")
    for spec in specs:
        info = export_nc_spec(
            spec,
            data_dir=data_dir,
            anygraph_root=anygraph_root,
            seed=args.seed,
        )
        print(
            f"  {info['slug']}: n={info['n_nodes']} e={info['n_edges']} "
            f"C={info['n_classes']} F={info['feat_dim']}"
        )
    print("Done.")


if __name__ == "__main__":
    main()
