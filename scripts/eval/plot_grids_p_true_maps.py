"""Plot Grids p(true class) maps collected by collect_grids_p_true_maps.py.

Reads ``<sweep_dir>/*/maps.npz`` + ``meta.json`` and writes walk-length and
train-free-filter panel figures plus a text summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _load_cells(sweep_dir: Path) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for meta_path in sorted(sweep_dir.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        maps_path = meta_path.parent / "maps.npz"
        if not maps_path.is_file():
            continue
        payload = np.load(maps_path)
        meta["_dir"] = str(meta_path.parent)
        meta["_p_true"] = payload["p_true"]
        meta["_pred"] = payload["pred"]
        meta["_y"] = payload["y"]
        meta["_train_mask"] = payload["train_mask"].astype(bool)
        meta["_grid_id"] = payload["grid_id"]
        meta["_hops"] = payload["hops"]
        cells.append(meta)
    return cells


def _fmt_acc(val: Optional[float]) -> str:
    if val is None:
        return "NA"
    return f"{100.0 * float(val):.2f}%"


def _grid_field(cell: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_true = cell["_p_true"]
    grid_id = cell["_grid_id"]
    train_mask = cell["_train_mask"]
    gid = int(cell.get("viz_grid", 0))
    h = int(cell.get("grid_h", 25))
    w = int(cell.get("grid_w", 25))
    n_per = h * w
    ids = np.flatnonzero(grid_id == gid)
    field = np.full((h, w), np.nan, dtype=np.float64)
    locals_ = ids % n_per
    field[locals_ // w, locals_ % w] = p_true[ids]
    train_ids = np.flatnonzero((grid_id == gid) & train_mask)
    train_local = train_ids % n_per
    return field, train_local // w, train_local % w


def _draw_grid(ax, cell: dict[str, Any], title: str, *, vmin: float = 0.0, vmax: float = 1.0):
    from matplotlib.colors import Normalize

    field, tr, tc = _grid_field(cell)
    im = ax.imshow(
        field,
        origin="upper",
        cmap="viridis",
        norm=Normalize(vmin=vmin, vmax=vmax),
        interpolation="nearest",
    )
    if tr.size:
        ax.scatter(
            tc, tr, marker="*", s=160, c="red", edgecolors="white", linewidths=0.7, zorder=3,
        )
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def _panel(cells: list[dict[str, Any]], titles: list[str], out_path: Path, *, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib not available; skip {out_path}")
        return
    if not cells:
        return
    n = len(cells)
    fig, axes = plt.subplots(
        1, n, figsize=(3.3 * n, 3.8), squeeze=False, constrained_layout=True,
    )
    im = None
    for ax, cell, lab in zip(axes[0], cells, titles):
        im = _draw_grid(ax, cell, lab)
        ax.set_aspect("equal")
    if im is not None:
        cbar = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02)
        cbar.set_label("p(correct class)")
    fig.suptitle(title, fontsize=12)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def _write_summary(cells: list[dict[str, Any]], path: Path) -> None:
    lines = ["Grids p(true class) maps", ""]
    for cell in cells:
        keep_p = cell.get("keep_train_free_p")
        filt = cell.get("filter_stats") or {}
        lines.append(
            f"{cell['run_name']}: walk_num={cell['walk_num']} walk_len={cell['walk_len']} "
            f"keep_p={keep_p} test_acc={_fmt_acc(cell.get('test_accuracy'))} "
            f"mean_p_true={cell.get('mean_p_true_query')}"
        )
        hop = cell.get("mean_p_true_by_hop") or {}
        if hop:
            parts = [f"{k}:{float(v):.3f}" for k, v in sorted(hop.items(), key=lambda kv: int(kv[0]))]
            lines.append("  p_true by hop: " + " ".join(parts))
        if filt:
            lines.append(
                "  walks sampled/kept/hit/free_kept/padded: "
                f"{filt.get('n_walks_sampled')} / {filt.get('n_walks_kept')} / "
                f"{filt.get('n_walks_hit_train')} / {filt.get('n_walks_train_free_kept')} / "
                f"{filt.get('n_walks_padded')}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_dir", type=str, required=True)
    args = parser.parse_args()
    sweep_dir = Path(args.sweep_dir)
    cells = _load_cells(sweep_dir)
    if not cells:
        raise SystemExit(f"no completed cells under {sweep_dir}")

    walk_len_cells = [c for c in cells if c.get("keep_train_free_p") is None]
    walk_len_cells.sort(key=lambda c: (int(c["walk_num"]), int(c["walk_len"])))
    filter_cells = [c for c in cells if c.get("keep_train_free_p") is not None]
    filter_cells.sort(key=lambda c: (float(c["keep_train_free_p"]), int(c["walk_num"])))

    if walk_len_cells:
        titles = [f"L={c['walk_len']}" for c in walk_len_cells]
        n = walk_len_cells[0]["walk_num"]
        gid = walk_len_cells[0].get("viz_grid", 0)
        _panel(
            walk_len_cells,
            titles,
            sweep_dir / "walk_len_panel.png",
            title=f"Grids p(correct class), walk_num={n}, grid={gid}",
        )
    if filter_cells:
        titles = [f"p={c['keep_train_free_p']}" for c in filter_cells]
        n = filter_cells[0]["walk_num"]
        l = filter_cells[0]["walk_len"]
        gid = filter_cells[0].get("viz_grid", 0)
        _panel(
            filter_cells,
            titles,
            sweep_dir / "train_free_filter_panel.png",
            title=f"Grids p(correct class), {n}x{l} 2x walks, keep train-free with p, grid={gid}",
        )
    _write_summary(cells, sweep_dir / "grids_p_true_maps_summary.txt")


if __name__ == "__main__":
    main()
