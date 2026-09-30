"""Launch GraphAny on one Wander NC slug (released checkpoint, protocol splits)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_WANDER_ROOT = Path(__file__).resolve().parents[2]
if str(_WANDER_ROOT) not in sys.path:
    sys.path.insert(0, str(_WANDER_ROOT))

from baselines.export_nc import add_nofeat_flags  # noqa: E402
from baselines.graphany.evaluate import dump_report, evaluate_slug  # noqa: E402
from baselines.paths import (  # noqa: E402
    WANDER_ROOT,
    require_checkout,
    resolve_graphany_data_root,
)
from baselines.registry import nc_eval_specs, resolve_nc_eval_spec  # noqa: E402
from data.nc_splits import (  # noqa: E402
    SPLIT_DIR_PREFIX,
    eval_nc_split_indices,
    n_eval_nc_splits,
)


def reports_root(slug: str, *, nofeat: bool) -> Path:
    folder = "nc-nofeat" if nofeat else "nc"
    return WANDER_ROOT / "results" / "baselines" / "graphany" / folder / slug


def report_path(slug: str, registry_key: str, split_index: int, *, nofeat: bool) -> Path:
    root = reports_root(slug, nofeat=nofeat)
    if n_eval_nc_splits(registry_key) <= 1:
        return root / "report.json"
    return root / f"{SPLIT_DIR_PREFIX}{int(split_index)}" / "report.json"


def _split_indices(registry_key: str, *, split: int | None, mode: str) -> list[int]:
    n = n_eval_nc_splits(registry_key)
    if split is not None:
        if split < 0 or split >= n:
            raise SystemExit(
                f"split={split} out of range for {registry_key} ({n} protocol splits)"
            )
        return [split]
    return eval_nc_split_indices(registry_key, mode=mode)


def _ckpt_key(raw: str) -> str:
    key = raw.strip().lower().removesuffix(".pt")
    if key.startswith("graph_any_"):
        key = key[len("graph_any_") :]
    return key


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GraphAny node classification on Wander protocol splits."
    )
    parser.add_argument("--dataset", default="", help="Slug or registry key, e.g. cora")
    parser.add_argument("--list", action="store_true", help="Print NC slugs and split counts.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument(
        "--ckpt",
        default=os.environ.get("GRAPHANY_CKPT", "arxiv"),
        help="arxiv (default), cora, wisconsin, product, or a .pt filename.",
    )
    parser.add_argument(
        "--split",
        type=int,
        default=None,
        help="Run one Wander NC protocol split index (default: every protocol split).",
    )
    parser.add_argument(
        "--split-mode",
        choices=("all", "first"),
        default=os.environ.get("GRAPHANY_SPLIT_MODE", "all"),
        help="all = official / GraphAny protocol splits; first = split 0 only.",
    )
    add_nofeat_flags(parser)
    args = parser.parse_args()

    if args.list:
        for nc_spec in nc_eval_specs(nofeat=True if args.nofeat else False):
            n = n_eval_nc_splits(nc_spec.registry_key)
            print(f"{nc_spec.slug}\t{nc_spec.registry_key}\tsplits={n}")
        return

    if not args.dataset.strip():
        raise SystemExit("Pass --dataset <name> (or --list).")

    if args.split is None:
        raw_split = os.environ.get("GRAPHANY_SPLIT", "").strip()
        if raw_split:
            args.split = int(raw_split)

    root = require_checkout("graphany")
    spec = resolve_nc_eval_spec(args.dataset.strip(), nofeat=args.nofeat)
    data_root = resolve_graphany_data_root()
    bundle = data_root / spec.slug
    has_split = (bundle / "split.npz").is_file() or (bundle / "split_0.npz").is_file()
    if not (bundle / "features.npy").is_file() or not has_split:
        raise SystemExit(
            f"Missing GraphAny bundle {bundle}. Run:\n"
            f"  python baselines/graphany/prepare.py --dataset {spec.slug}"
        )

    ckpt_key = _ckpt_key(args.ckpt)
    cpu = bool(args.cpu) or os.environ.get("GRAPHANY_CPU", "0") == "1"
    indices = _split_indices(spec.registry_key, split=args.split, mode=args.split_mode)
    n_protocol = n_eval_nc_splits(spec.registry_key)
    cache_dir = root / "data_cache" / "wander" / spec.slug
    print(
        f"GraphAny slug={spec.slug} ckpt={ckpt_key} "
        f"splits={indices}/{n_protocol} data={bundle}"
    )

    for split_index in indices:
        print(f"--- split {split_index}/{n_protocol} ---")
        report = evaluate_slug(
            graphany_root=root,
            slug=spec.slug,
            registry_key=spec.registry_key,
            data_root=data_root,
            ckpt=ckpt_key,
            cache_dir=cache_dir,
            split_index=split_index,
            cpu=cpu,
        )
        out = report_path(
            spec.slug, spec.registry_key, split_index, nofeat=spec.ignore_features
        )
        dump_report(out, report)
        test = report.get("metrics", {}).get("test", {})
        print(
            f"Wrote {out} status={report.get('status')} "
            f"test_acc={test.get('accuracy')} test_auroc={test.get('roc-auc')}"
        )


if __name__ == "__main__":
    main()
