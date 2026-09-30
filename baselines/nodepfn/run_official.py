"""Launch official NodePFN inference with the released checkpoint (paper splits)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_WANDER_ROOT = Path(__file__).resolve().parents[2]
if str(_WANDER_ROOT) not in sys.path:
    sys.path.insert(0, str(_WANDER_ROOT))

from baselines.nodepfn.configs import NODEPFN_RUN_SPECS, spec_by_dataset  # noqa: E402
from baselines.paths import require_checkout  # noqa: E402


def _cmd_for_spec(spec, *, base_model_path: str) -> list[str]:
    cmd = [
        sys.executable,
        "nodepfn/node_classification.py",
        "--dataset",
        spec.dataset,
        "--base_model_path",
        base_model_path,
        "--dim_reduction",
        spec.dim_reduction,
        "--n_components",
        str(spec.n_components),
        "--runs",
        str(spec.runs),
        "--smoothing_steps",
        str(spec.smoothing_steps),
    ]
    if spec.n_ensemble is not None:
        cmd.extend(["--n_ensemble", str(spec.n_ensemble)])
    if spec.svd_algorithm is not None:
        cmd.extend(["--svd_algorithm", spec.svd_algorithm])
    if spec.label_num_per_class is not None:
        cmd.extend(["--label_num_per_class", str(spec.label_num_per_class)])
    if spec.cpu:
        cmd.append("--cpu")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run NodePFN on official datasets/splits (paper protocol)."
    )
    parser.add_argument("--dataset", default="", help="One dataset name, or omit to list.")
    parser.add_argument(
        "--base_model_path",
        default="models_ckpts/nodepfn",
        help="Directory containing checkpoint_epoch_30.ckpt (relative to NodePFN root).",
    )
    parser.add_argument("--list", action="store_true", help="Print supported datasets and exit.")
    args = parser.parse_args()

    root = require_checkout("nodepfn")
    if args.list:
        for spec in NODEPFN_RUN_SPECS:
            print(spec.dataset)
        return
    if not args.dataset:
        raise SystemExit("Pass --dataset <name> (or --list).")

    spec = spec_by_dataset(args.dataset)
    cmd = _cmd_for_spec(spec, base_model_path=args.base_model_path)
    print(f"NodePFN cwd={root} cmd={' '.join(cmd)}")
    raise SystemExit(subprocess.call(cmd, cwd=str(root), env=os.environ.copy()))


if __name__ == "__main__":
    main()
