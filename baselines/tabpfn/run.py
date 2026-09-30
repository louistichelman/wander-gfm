"""Launch TabPFNv2 on one Wander NC slug (tabular features; default PCA 64)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_WANDER_ROOT = Path(__file__).resolve().parents[2]
if str(_WANDER_ROOT) not in sys.path:
    sys.path.insert(0, str(_WANDER_ROOT))

from baselines.export_nc import add_nofeat_flags  # noqa: E402
from baselines.paths import WANDER_ROOT, resolve_tabpfn_bundle_dir  # noqa: E402
from baselines.registry import nc_eval_specs, resolve_nc_eval_spec  # noqa: E402
from baselines.tabpfn.configs import (  # noqa: E402
    DEFAULT_FIT_MODE,
    DEFAULT_MODEL_VERSION,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_PCA_DIM,
    DEFAULT_SEED,
)
from baselines.tabpfn.evaluate import dump_report, evaluate_slug  # noqa: E402
from data.nc_splits import (  # noqa: E402
    SPLIT_DIR_PREFIX,
    eval_nc_split_indices,
    n_eval_nc_splits,
)


def reports_root(slug: str, *, nofeat: bool) -> Path:
    folder = "nc-nofeat" if nofeat else "nc"
    return WANDER_ROOT / "results" / "baselines" / "tabpfn" / folder / slug


def report_path(
    slug: str,
    registry_key: str,
    split_index: int,
    *,
    nofeat: bool,
) -> Path:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run TabPFNv2 node classification on Wander protocol splits "
            "(ignore graph; default StandardScaler + PCA 64)."
        )
    )
    parser.add_argument("--dataset", default="", help="Slug or registry key, e.g. cora")
    parser.add_argument("--list", action="store_true", help="Print NC-eval slugs.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=int(os.environ.get("TABPFN_PCA_DIM", str(DEFAULT_PCA_DIM))),
        help=f"PCA target dim (default {DEFAULT_PCA_DIM}; 0 disables PCA).",
    )
    parser.add_argument(
        "--model-version",
        default=os.environ.get("TABPFN_MODEL_VERSION", DEFAULT_MODEL_VERSION),
        choices=("v2", "v2.5", "v2.6", "v3"),
        help="TabPFN checkpoint version (default v2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("TABPFN_SEED", str(DEFAULT_SEED))),
        help="RNG seed for PCA / TabPFN.",
    )
    parser.add_argument(
        "--fit-mode",
        default=os.environ.get("TABPFN_FIT_MODE", DEFAULT_FIT_MODE),
        help="TabPFN fit_mode (see TabPFNClassifier docs).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=int(os.environ.get("TABPFN_N_ESTIMATORS", str(DEFAULT_N_ESTIMATORS))),
        help="TabPFN ensemble size (0 = package default 'auto').",
    )
    parser.add_argument(
        "--respect-pretraining-limits",
        action="store_true",
        help="Do not pass ignore_pretraining_limits=True (default ignores limits).",
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
        default=os.environ.get("TABPFN_SPLIT_MODE", "all"),
        help="all = official / GraphAny protocol splits; first = split 0 only.",
    )
    add_nofeat_flags(parser)
    args = parser.parse_args()

    if args.list:
        for nc_spec in nc_eval_specs(nofeat=True if args.nofeat else False):
            n = n_eval_nc_splits(nc_spec.registry_key)
            print(f"{nc_spec.slug}\tsplits={n}\tpca={args.pca_dim}")
        return

    if not args.dataset.strip():
        raise SystemExit("Pass --dataset <name> (or --list).")

    if args.split is None:
        raw_split = os.environ.get("TABPFN_SPLIT", "").strip()
        if raw_split:
            args.split = int(raw_split)

    spec = resolve_nc_eval_spec(args.dataset.strip(), nofeat=args.nofeat)
    bundle = resolve_tabpfn_bundle_dir(spec.slug)
    has_split = (bundle / "split.npz").is_file() or (bundle / "split_0.npz").is_file()
    if not (bundle / "features.npy").is_file() or not has_split:
        raise SystemExit(
            f"Missing TabPFN bundle {bundle}. Run:\n"
            f"  python baselines/tabpfn/prepare.py --dataset {spec.slug}\n"
            f"Or point TABPFN_DATA_ROOT at an existing nodepfn_nc tree."
        )

    cpu = bool(args.cpu) or os.environ.get("TABPFN_CPU", "0") == "1"
    ignore_limits = not bool(args.respect_pretraining_limits)
    if os.environ.get("TABPFN_RESPECT_LIMITS", "0") == "1":
        ignore_limits = False
    indices = _split_indices(spec.registry_key, split=args.split, mode=args.split_mode)
    n_protocol = n_eval_nc_splits(spec.registry_key)

    print(
        f"TabPFN slug={spec.slug} version={args.model_version} pca_dim={args.pca_dim} "
        f"ignore_graph=True splits={indices}/{n_protocol} "
        f"device={'cpu' if cpu else 'cuda'} data={bundle}"
    )

    last_status = None
    for split_index in indices:
        print(f"--- split {split_index}/{n_protocol} ---")
        report = evaluate_slug(
            slug=spec.slug,
            bundle_dir=bundle,
            registry_key=spec.registry_key,
            split_index=split_index,
            pca_dim=int(args.pca_dim),
            model_version=str(args.model_version),
            seed=int(args.seed),
            cpu=cpu,
            ignore_pretraining_limits=ignore_limits,
            fit_mode=str(args.fit_mode),
            n_estimators=int(args.n_estimators),
        )
        out = report_path(
            spec.slug,
            spec.registry_key,
            split_index,
            nofeat=spec.ignore_features,
        )
        dump_report(out, report)
        test = report.get("metrics", {}).get("test", {})
        print(
            f"Wrote {out} status={report.get('status')} "
            f"test_acc={test.get('accuracy')} test_auroc={test.get('roc-auc')}"
        )
        last_status = report.get("status")
        if last_status == "skipped":
            print(f"skipped: {report.get('reason')}")
            break
        if last_status == "error":
            print(f"error: {report.get('reason')}")
            break

    if last_status in {"error"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
