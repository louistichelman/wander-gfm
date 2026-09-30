"""Launch official AnyGraph zero-shot eval (pretrained checkpoint, epoch 0)."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

_WANDER_ROOT = Path(__file__).resolve().parents[2]
if str(_WANDER_ROOT) not in sys.path:
    sys.path.insert(0, str(_WANDER_ROOT))

from baselines.export_nc import add_nofeat_flags  # noqa: E402
from baselines.paths import (  # noqa: E402
    WANDER_ROOT,
    require_checkout,
    resolve_anygraph_data_root,
    resolve_anygraph_nc_data_root,
)
from baselines.registry import resolve_nc_eval_spec  # noqa: E402

_ACC_RE = re.compile(
    r"(?P<slug>\S+)\s+Test:.*?Acc_mean\s*=\s*(?P<acc>[-0-9.]+).*?F1_mean\s*=\s*(?P<f1>[-0-9.]+)",
    re.S,
)
_ACC_STD_RE = re.compile(r"Acc_std\s*=\s*(?P<v>[-0-9.]+)")
_F1_STD_RE = re.compile(r"F1_std\s*=\s*(?P<v>[-0-9.]+)")


def nc_report_path(slug: str, *, nofeat: bool) -> Path:
    folder = "nc-nofeat" if nofeat else "nc"
    return WANDER_ROOT / "results" / "baselines" / "anygraph" / folder / slug / "report.json"


def _parse_nc_log(text: str, slug: str) -> dict:
    matches = list(_ACC_RE.finditer(text))
    chosen = None
    for match in matches:
        if match.group("slug") == slug:
            chosen = match
    if chosen is None and matches:
        chosen = matches[-1]
    if chosen is None:
        return {}
    chunk = chosen.group(0)
    acc_std = _ACC_STD_RE.search(chunk)
    f1_std = _F1_STD_RE.search(chunk)
    return {
        "accuracy": float(chosen.group("acc")),
        "f1": float(chosen.group("f1")),
        "accuracy_std": float(acc_std.group("v")) if acc_std else 0.0,
        "f1_std": float(f1_std.group("v")) if f1_std else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AnyGraph zero-shot evaluation.")
    parser.add_argument("--load_model", default="", help="e.g. pretrain_link1 / pretrain_link2")
    parser.add_argument("--dataset_setting", default="", help="e.g. link2 or a single NC slug")
    parser.add_argument("--dataset", default="", help="Wander NC slug (implies --eval_type node).")
    parser.add_argument("--eval_type", choices=["link", "node"], default="link")
    parser.add_argument(
        "--eval_protocol_wander",
        action="store_true",
        default=os.environ.get("EVAL_PROTOCOL_WANDER", "0") == "1",
        help="Homogeneous Recall@K: drop reverse-in-train queries and symmetrize train mask.",
    )
    parser.add_argument("--gpu", default="0")
    add_nofeat_flags(parser)
    args = parser.parse_args()

    spec = None
    if args.dataset.strip():
        spec = resolve_nc_eval_spec(args.dataset.strip(), nofeat=args.nofeat)
        args.eval_type = "node"
        args.dataset_setting = spec.slug
        if not args.load_model:
            args.load_model = "pretrain_link2"
    if not args.dataset_setting:
        raise SystemExit("Pass --dataset_setting or --dataset.")
    if not args.load_model:
        raise SystemExit("Pass --load_model (e.g. pretrain_link2).")

    root = require_checkout("anygraph")
    cmd = [
        sys.executable,
        "main.py",
        "--load_model",
        args.load_model,
        "--epoch",
        "0",
        "--dataset_setting",
        args.dataset_setting,
        "--gpu",
        args.gpu,
    ]
    if args.eval_protocol_wander and args.eval_type == "link":
        cmd.append("--eval_protocol_wander")
    if args.eval_type == "node" and spec is not None and spec.ignore_features:
        cmd.extend(["--proj_method", "adj_svd"])

    cwd = root / "node_classification" if args.eval_type == "node" else root
    env = os.environ.copy()
    if args.eval_type == "node" and spec is not None:
        env["ANYGRAPH_DATA_ROOT"] = str(resolve_anygraph_nc_data_root())
    else:
        env.setdefault("ANYGRAPH_DATA_ROOT", str(resolve_anygraph_data_root()))
    print(
        f"AnyGraph cwd={cwd} data={env['ANYGRAPH_DATA_ROOT']} cmd={' '.join(cmd)}"
    )
    if spec is not None:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)
        sys.stdout.write(proc.stdout or "")
        sys.stderr.write(proc.stderr or "")
        parsed = _parse_nc_log((proc.stdout or "") + "\n" + (proc.stderr or ""), spec.slug)
        report = {
            "function": "baselines.anygraph.run",
            "slug": spec.slug,
            "checkpoint": args.load_model,
            "metrics": {"test": parsed},
        }
        out = nc_report_path(spec.slug, nofeat=spec.ignore_features)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote {out} {parsed}")
        raise SystemExit(proc.returncode)
    raise SystemExit(subprocess.call(cmd, cwd=str(cwd), env=env))


if __name__ == "__main__":
    main()
