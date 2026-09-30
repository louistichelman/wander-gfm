"""Launch GraphPFN ICL evaluation for one dataset slug (all protocol splits)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_WANDER_ROOT = Path(__file__).resolve().parents[2]
if str(_WANDER_ROOT) not in sys.path:
    sys.path.insert(0, str(_WANDER_ROOT))

from baselines.graphpfn.config_paths import (  # noqa: E402
    config_relpath_for_finetune,
    icl_eval_relpath,
    icl_n_seeds_for_spec,
    resolve_pca_dim,
    resolve_run_label,
)
from baselines.paths import GRAPHPFN_ROOT, require_checkout  # noqa: E402
from baselines.registry import spec_by_slug  # noqa: E402
from data.nc_splits import eval_nc_split_indices, n_eval_nc_splits  # noqa: E402


def _split_indices(spec, *, split: int | None, mode: str) -> list[int]:
    if split is not None:
        n = n_eval_nc_splits(spec.registry_key)
        if split < 0 or split >= n:
            raise SystemExit(
                f"split={split} out of range for {spec.registry_key} ({n} protocol splits)"
            )
        return [split]
    return eval_nc_split_indices(spec.registry_key, mode=mode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GraphPFN ICL on one dataset.")
    parser.add_argument("--dataset", required=True, help="GraphPFN slug, e.g. cora")
    parser.add_argument("--pca-dim", default=os.environ.get("GRAPHPFN_PCA_DIM", "64"))
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=None,
        help="Ensemble seeds per data split (default: 1 if several protocol splits, else 10).",
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
        default=os.environ.get("GRAPHPFN_SPLIT_MODE", "all"),
        help="all = official / GraphAny protocol splits; first = split 0 only.",
    )
    parser.add_argument("--force", action="store_true", default=os.environ.get("FORCE", "1") == "1")
    parser.add_argument("--no-force", dest="force", action="store_false")
    parser.add_argument(
        "--finetune",
        action="store_true",
        default=os.environ.get("GRAPHPFN_FINETUNE", "").strip().lower() in {"1", "true", "yes"},
        help="Run LR-grid finetune (tuning.toml) instead of ICL evaluation.toml.",
    )
    args = parser.parse_args()
    if args.split is None:
        raw_split = os.environ.get("GRAPHPFN_SPLIT", "").strip()
        if raw_split:
            args.split = int(raw_split)

    require_checkout("graphpfn")
    spec = spec_by_slug(args.dataset)
    pca_dim = resolve_pca_dim(pca_dim=args.pca_dim)
    env_seeds = os.environ.get("N_SEEDS")
    n_seeds_override = args.n_seeds
    if n_seeds_override is None and env_seeds:
        n_seeds_override = int(env_seeds)
    n_seeds = icl_n_seeds_for_spec(spec, n_seeds_override)
    label = resolve_run_label(pca_dim=args.pca_dim, ignore_features=spec.ignore_features)

    if args.finetune:
        config_rel = (
            f"{config_relpath_for_finetune(pca_dim, ignore_features=spec.ignore_features)}"
            f"/{spec.slug}/tuning.toml"
        )
        generate_hint = "python baselines/graphpfn/generate_finetune_configs.py --pca-dim 64"
        config = GRAPHPFN_ROOT / config_rel
        if not config.is_file():
            raise SystemExit(f"Missing config {config}. Generate with:\n  {generate_hint}")
        print(f"GraphPFN finetune slug={spec.slug} label={label} config={config_rel}")
        cmd = [sys.executable, "bin/go.py", config_rel, "--n_seeds", str(n_seeds)]
        if args.force:
            cmd.append("--force")
        env = os.environ.copy()
        env.setdefault("DGLBACKEND", "pytorch")
        pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(GRAPHPFN_ROOT) + (os.pathsep + pythonpath if pythonpath else "")
        raise SystemExit(subprocess.call(cmd, cwd=str(GRAPHPFN_ROOT), env=env))

    generate_hint = "python baselines/graphpfn/generate_icl_configs.py"
    if spec.ignore_features:
        generate_hint += " --nofeat"
    indices = _split_indices(spec, split=args.split, mode=args.split_mode)
    n_protocol = n_eval_nc_splits(spec.registry_key)
    env = os.environ.copy()
    env.setdefault("DGLBACKEND", "pytorch")
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(GRAPHPFN_ROOT) + (os.pathsep + pythonpath if pythonpath else "")

    exit_code = 0
    for split_index in indices:
        config_rel = icl_eval_relpath(spec, pca_dim, split_index)
        config = GRAPHPFN_ROOT / config_rel
        if not config.is_file():
            raise SystemExit(f"Missing config {config}. Generate with:\n  {generate_hint}")
        print(
            f"GraphPFN ICL slug={spec.slug} split={split_index}/{n_protocol} "
            f"n_seeds={n_seeds} label={label} config={config_rel}"
        )
        cmd = [sys.executable, "bin/go.py", config_rel, "--n_seeds", str(n_seeds)]
        if args.force:
            cmd.append("--force")
        rc = subprocess.call(cmd, cwd=str(GRAPHPFN_ROOT), env=env)
        if rc != 0:
            exit_code = rc
            break
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
