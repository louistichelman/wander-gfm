"""Launch NodePFN on one Wander NC slug (released checkpoint, protocol splits)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_WANDER_ROOT = Path(__file__).resolve().parents[2]
if str(_WANDER_ROOT) not in sys.path:
    sys.path.insert(0, str(_WANDER_ROOT))

from baselines.export_nc import add_nofeat_flags  # noqa: E402
from baselines.nodepfn.configs import (  # noqa: E402
    GRID_SMOOTHING_STEPS,
    GRID_TSVD_COMPONENTS,
    NodePFNRunSpec,
    apply_tsvd_grid_spec,
    featured_slugs_for_tsvd_grid,
    grid_cell_tag,
    inference_runs,
    spec_for_nc_slug,
)
from baselines.nodepfn.evaluate import dump_report, evaluate_slug  # noqa: E402
from baselines.paths import (  # noqa: E402
    WANDER_ROOT,
    require_checkout,
    resolve_nodepfn_data_root,
)
from baselines.registry import nc_eval_specs, resolve_nc_eval_spec  # noqa: E402
from data.nc_splits import (  # noqa: E402
    SPLIT_DIR_PREFIX,
    eval_nc_split_indices,
    n_eval_nc_splits,
)


def reports_root(slug: str, *, nofeat: bool, grid_tag: str | None = None) -> Path:
    if grid_tag:
        return WANDER_ROOT / "results" / "baselines" / "nodepfn" / "nc-grid" / slug / grid_tag
    folder = "nc-nofeat" if nofeat else "nc"
    return WANDER_ROOT / "results" / "baselines" / "nodepfn" / folder / slug


def report_path(
    slug: str,
    registry_key: str,
    split_index: int,
    *,
    nofeat: bool,
    grid_tag: str | None = None,
) -> Path:
    root = reports_root(slug, nofeat=nofeat, grid_tag=grid_tag)
    if n_eval_nc_splits(registry_key) <= 1:
        return root / "report.json"
    return root / f"{SPLIT_DIR_PREFIX}{int(split_index)}" / "report.json"


def _grid_cell_done(
    slug: str,
    registry_key: str,
    indices: list[int],
    *,
    nofeat: bool,
    grid_tag: str,
) -> bool:
    import json

    for split_index in indices:
        path = report_path(
            slug, registry_key, split_index, nofeat=nofeat, grid_tag=grid_tag
        )
        if not path.is_file():
            return False
        try:
            report = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        if report.get("status") != "ok":
            return False
    return True


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
        description="Run NodePFN node classification on Wander protocol splits."
    )
    parser.add_argument("--dataset", default="", help="Slug or registry key, e.g. cora")
    parser.add_argument("--list", action="store_true", help="Print NC slugs and hparams.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument("--runs", type=int, default=0, help="Override TSVD/ensemble run count.")
    parser.add_argument(
        "--batch-size-inference",
        type=int,
        default=int(os.environ.get("NODEPFN_BATCH_SIZE_INFERENCE", "32")),
        help="NodePFN ensemble chunk size (lower this on dense graphs to avoid CUDA OOM).",
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
        default=os.environ.get("NODEPFN_SPLIT_MODE", "all"),
        help="all = official / GraphAny protocol splits; first = split 0 only.",
    )
    parser.add_argument(
        "--base_model_path",
        default="",
        help="Directory containing checkpoint_epoch_30.ckpt (default: NodePFN models_ckpts/nodepfn).",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=0,
        help="Override TSVD components (0 = spec default).",
    )
    parser.add_argument(
        "--smoothing-steps",
        type=int,
        default=-1,
        help="Override SimpleConv steps (-1 = spec default).",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Run TSVD {10,15,20,25,30} × smoothing {0,1}; write under nc-grid/.",
    )
    add_nofeat_flags(parser)
    args = parser.parse_args()

    if args.list:
        if args.grid:
            print("\n".join(featured_slugs_for_tsvd_grid()))
            return
        for nc_spec in nc_eval_specs(nofeat=True if args.nofeat else False):
            h = spec_for_nc_slug(nc_spec.slug)
            n = n_eval_nc_splits(nc_spec.registry_key)
            runs = inference_runs(nc_spec.registry_key, h)
            print(
                f"{nc_spec.slug}\t{h.hparams_source}\tsmooth={h.smoothing_steps}\t"
                f"red={h.dim_reduction}:{h.n_components}\t"
                f"ens={h.n_ensemble}\truns={runs}\tsplits={n}"
            )
        return

    if not args.dataset.strip():
        raise SystemExit("Pass --dataset <name> (or --list).")

    if args.split is None:
        raw_split = os.environ.get("NODEPFN_SPLIT", "").strip()
        if raw_split:
            args.split = int(raw_split)

    require_checkout("nodepfn")
    spec = resolve_nc_eval_spec(args.dataset.strip(), nofeat=args.nofeat)
    data_root = resolve_nodepfn_data_root()
    bundle = data_root / spec.slug
    has_split = (bundle / "split.npz").is_file() or (bundle / "split_0.npz").is_file()
    if not (bundle / "features.npy").is_file() or not has_split:
        raise SystemExit(
            f"Missing NodePFN bundle {bundle}. Run:\n"
            f"  python baselines/nodepfn/prepare.py --dataset {spec.slug}"
        )

    base_hparams = spec_for_nc_slug(spec.slug)
    ckpt = None
    if args.base_model_path:
        ckpt = Path(args.base_model_path)
        if not ckpt.is_absolute():
            ckpt = (require_checkout("nodepfn") / ckpt).resolve()
    cpu = bool(args.cpu) or os.environ.get("NODEPFN_CPU", "0") == "1"
    runs_override = args.runs if args.runs > 0 else None
    batch_size = max(1, int(args.batch_size_inference))
    grid = bool(args.grid) or os.environ.get("NODEPFN_GRID", "0") == "1"
    indices = _split_indices(spec.registry_key, split=args.split, mode=args.split_mode)
    n_protocol = n_eval_nc_splits(spec.registry_key)

    cells: list[tuple[NodePFNRunSpec, str | None]]
    if grid:
        cells = [
            (
                apply_tsvd_grid_spec(base_hparams, k, s),
                grid_cell_tag(k, s),
            )
            for k in GRID_TSVD_COMPONENTS
            for s in GRID_SMOOTHING_STEPS
        ]
    else:
        hparams = base_hparams
        n_comp = args.n_components or int(os.environ.get("NODEPFN_N_COMPONENTS", "0") or 0)
        smooth_env = os.environ.get("NODEPFN_SMOOTHING_STEPS", "")
        smooth = args.smoothing_steps if args.smoothing_steps >= 0 else (
            int(smooth_env) if smooth_env.strip() else -1
        )
        tag = None
        if n_comp > 0 or smooth >= 0:
            k = n_comp if n_comp > 0 else int(hparams.n_components)
            s = smooth if smooth >= 0 else int(hparams.smoothing_steps)
            hparams = apply_tsvd_grid_spec(hparams, k, s)
            tag = grid_cell_tag(k, s)
        cells = [(hparams, tag)]

    last_status = None
    for hparams, tag in cells:
        n_runs = inference_runs(spec.registry_key, hparams, runs_override)
        print(
            f"NodePFN slug={spec.slug} source={hparams.hparams_source} "
            f"smooth={hparams.smoothing_steps} red={hparams.dim_reduction} "
            f"n={hparams.n_components} splits={indices}/{n_protocol} "
            f"runs_per_split={n_runs} batch_size_inference={batch_size} "
            f"query_attn_chunk={os.environ.get('NODEPFN_QUERY_ATTN_CHUNK', '4096')} "
            f"device={'cpu' if cpu else 'cuda'} data={bundle} grid_tag={tag}"
        )
        if tag and _grid_cell_done(
            spec.slug, spec.registry_key, indices, nofeat=spec.ignore_features, grid_tag=tag
        ):
            print(f"  skip {tag}: reports already present")
            last_status = "ok"
            continue
        for split_index in indices:
            print(f"--- {tag or 'default'} split {split_index}/{n_protocol} ---")
            report = evaluate_slug(
                slug=spec.slug,
                bundle_dir=bundle,
                registry_key=spec.registry_key,
                spec=hparams,
                checkpoint_dir=ckpt,
                cpu=cpu,
                runs=n_runs,
                batch_size_inference=batch_size,
                split_index=split_index,
            )
            out = report_path(
                spec.slug,
                spec.registry_key,
                split_index,
                nofeat=spec.ignore_features,
                grid_tag=tag,
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
        if last_status == "skipped":
            break


if __name__ == "__main__":
    main()
