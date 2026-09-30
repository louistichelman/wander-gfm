"""Launch official UniLP (context_LP) zero-shot inference."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_WANDER_ROOT = Path(__file__).resolve().parents[2]
if str(_WANDER_ROOT) not in sys.path:
    sys.path.insert(0, str(_WANDER_ROOT))

from baselines.paths import (  # noqa: E402
    WANDER_ROOT,
    require_checkout,
    resolve_unilp_checkpoint,
    resolve_unilp_data_root,
)

# Paper Table 19 graphs (Wander appendix). Official README also lists syn-*.
PAPER_INFERENCE_DATASETS = (
    "Celegans",
    "USAir",
    "PB",
    "NS",
    "Cora",
    "CS",
    "snap-musae-facebook",
)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, "").strip() or default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    return int(raw) if raw else default


def report_path() -> Path:
    return WANDER_ROOT / "results" / "baselines" / "unilp" / "report.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run official UniLP inference on the released checkpoint."
    )
    parser.add_argument(
        "--datasets",
        default=_env("UNILP_DATASETS", ",".join(PAPER_INFERENCE_DATASETS)),
        help="Comma-separated official UniLP names (default: Table 19 graphs).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=_env_int("UNILP_K", 200),
        help="In-context positive/negative links (paper inference: 200).",
    )
    parser.add_argument(
        "--load_model",
        default=_env("UNILP_CKPT"),
        help="Path to model.pt (default: third_party/context_LP/checkpoints/pretrained/model.pt).",
    )
    parser.add_argument(
        "--dataset_dir",
        default=_env("UNILP_DATA_DIR"),
        help="Official --dataset_dir (default: third_party/context_LP/data).",
    )
    parser.add_argument(
        "--log_dir",
        default="",
        help="Official --log_dir (default: results/baselines/unilp/logs).",
    )
    args, extra = parser.parse_known_args()

    root = require_checkout("unilp")
    ckpt = Path(args.load_model) if args.load_model else resolve_unilp_checkpoint()
    if not ckpt.is_file():
        raise SystemExit(
            f"Missing UniLP weights at {ckpt}. Place model.pt in "
            "third_party/context_LP/checkpoints/pretrained/ (see third_party/README.md)."
        )
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else resolve_unilp_data_root()
    log_dir = Path(args.log_dir) if args.log_dir else (
        WANDER_ROOT / "results" / "baselines" / "unilp" / "logs"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    datasets = ",".join(
        part.strip() for part in args.datasets.split(",") if part.strip()
    )
    if not datasets:
        raise SystemExit("Pass --datasets (comma-separated official UniLP names).")

    cmd = [
        sys.executable,
        "main.py",
        f"--inference_datasets={datasets}",
        f"--k={args.k}",
        f"--load_model={ckpt}",
        f"--dataset_dir={dataset_dir}",
        f"--log_dir={log_dir}",
        *extra,
    ]
    print(f"UniLP cwd={root} data={dataset_dir} ckpt={ckpt} cmd={' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(root), check=False)
    report = {
        "datasets": datasets,
        "k": int(args.k),
        "checkpoint": str(ckpt),
        "dataset_dir": str(dataset_dir),
        "log_dir": str(log_dir),
        "returncode": int(proc.returncode),
    }
    out = report_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {out}")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
