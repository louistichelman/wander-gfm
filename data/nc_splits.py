"""Node-classification eval splits: predefined multi-splits vs GraphAny-generated.

Wander NC eval protocol:
  1. Datasets that ship several official train/val/test masks: evaluate every
     split and report mean ± std.
  2. Datasets with a single official split: keep that split.
  3. Datasets with no official split: GraphAny ``get_data_split_masks``
     (20 labeled nodes per class, stratified, sklearn seed = split index),
     over 5 splits (indices 0..4). Split 0 matches the previous single-seed
     eval (experiment seed 0).

This module has no torch / dataset-registry import at module level so the
SLURM login-node coordinator can list split indices cheaply.
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

GRAPHANY_GENERATED_N_SPLITS = 5

# Official 2-D node-mask counts (PyG / Platonov). WikiCS shares one test mask
# across 20 train/val splits; the others have matching train/val/test columns.
PREDEFINED_NODE_SPLIT_COUNTS = {
    "ACTOR": 10,
    "AMAZON_RATINGS": 10,
    "CHAMELEON_FILTERED": 10,
    "CORNELL": 10,
    "ROMAN_EMPIRE": 10,
    "SQUIRREL_FILTERED": 10,
    "TEXAS": 10,
    "WISCONSIN": 10,
    "WIKI_CS": 20,
}

# HAS_PREDEFINED_NODE_SPLIT = False NC graphs: GraphAny 20-per-class splits.
GRAPHANY_GENERATED_DATASETS = frozenset(
    {
        "BRAZIL",
        "CO_CS",
        "CO_PHYSICS",
        "COMPUTERS",
        "EUROPE",
        "FULL_CORA",
        "FULL_DBLP",
        "PHOTO",
        "USA",
    }
)

SPLIT_DIR_PREFIX = "split_"


def n_eval_nc_splits(dataset_key: str) -> int:
    """How many NC eval splits the reporting protocol uses for ``dataset_key``."""
    key = dataset_key.upper()
    if key in PREDEFINED_NODE_SPLIT_COUNTS:
        return int(PREDEFINED_NODE_SPLIT_COUNTS[key])
    if key in GRAPHANY_GENERATED_DATASETS:
        return GRAPHANY_GENERATED_N_SPLITS
    return 1


def eval_nc_split_indices(
    dataset_key: str,
    *,
    mode: str = "all",
) -> list[int]:
    """Return split indices to run for ``dataset_key``.

    ``mode``:
      - ``all``: every protocol split
      - ``first``: only split 0 (legacy single-split eval)
    """
    n = n_eval_nc_splits(dataset_key)
    if mode == "first":
        return [0]
    if mode != "all":
        raise ValueError(f"mode must be 'all' or 'first', got {mode!r}")
    return list(range(n))


def split_run_name(
    dataset_key: str,
    split_index: int,
    n_splits: Optional[int] = None,
    *,
    stem: Optional[str] = None,
) -> str:
    """Checkpoint / metrics subfolder for one eval job.

    Single-split datasets keep ``DATASET`` (legacy layout). Multi-split jobs
    write ``DATASET/split_{i}``. ``stem`` overrides the folder name (GraphPFN
    slugs such as ``amazon-photo`` instead of ``PHOTO``).
    """
    name = dataset_key if stem is None else stem
    n = n_eval_nc_splits(dataset_key) if n_splits is None else int(n_splits)
    if n <= 1:
        return name
    return f"{name}/{SPLIT_DIR_PREFIX}{int(split_index)}"


def split_npz_filename(
    dataset_key: str,
    split_index: int,
    n_splits: Optional[int] = None,
) -> str:
    """On-disk split file for GraphPFN / npy exporters.

    Single-split datasets keep ``split.npz``. Multi-split jobs write
    ``split_{i}.npz`` (and exporters also copy split 0 to ``split.npz`` for
    legacy configs).
    """
    n = n_eval_nc_splits(dataset_key) if n_splits is None else int(n_splits)
    if n <= 1:
        return "split.npz"
    return f"{SPLIT_DIR_PREFIX}{int(split_index)}.npz"


def default_icl_n_seeds(dataset_key: str) -> int:
    """Ensemble seeds per data split: 1 when averaging protocol splits, else 10."""
    return 1 if n_eval_nc_splits(dataset_key) > 1 else 10


def parse_split_dir_name(name: str) -> Optional[int]:
    """Return the split index if ``name`` is ``split_{i}``, else None."""
    if not name.startswith(SPLIT_DIR_PREFIX):
        return None
    suffix = name[len(SPLIT_DIR_PREFIX) :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def n_mask_splits(mask, num_nodes: int) -> int:
    """Number of official splits stored in a 1-D or 2-D boolean mask tensor."""
    if mask is None:
        return 1
    ndim = int(mask.ndim)
    if ndim == 1:
        return 1
    if ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D split mask, got ndim={ndim}")
    shape = tuple(int(s) for s in mask.shape)
    if shape[0] == num_nodes:
        return shape[1]
    if shape[1] == num_nodes:
        return shape[0]
    raise ValueError(
        f"Mask shape {shape} is incompatible with num_nodes={num_nodes}"
    )


def n_predefined_splits_from_masks(
    train_mask,
    val_mask,
    test_mask,
    num_nodes: int,
) -> int:
    """Max split count across train/val/test (WikiCS test is 1-D, train is 20)."""
    counts = [
        n_mask_splits(mask, num_nodes)
        for mask in (train_mask, val_mask, test_mask)
        if mask is not None
    ]
    return max(counts) if counts else 1


def select_node_split_mask(mask, split_index: int, num_nodes: int):
    """Pick one 1-D ``[num_nodes]`` mask from a 1-D or 2-D official split tensor.

    1-D masks (Planetoid, GraphLand, WikiCS test) are returned as-is for every
    ``split_index``. 2-D masks must contain ``split_index``.
    """
    if mask is None:
        raise ValueError("split mask is None")
    ndim = int(mask.ndim)
    if ndim == 1:
        if int(mask.numel()) != num_nodes:
            raise ValueError(
                f"1-D mask length {int(mask.numel())} does not match "
                f"num_nodes={num_nodes}"
            )
        return mask
    n_splits = n_mask_splits(mask, num_nodes)
    if split_index < 0 or split_index >= n_splits:
        raise IndexError(
            f"split_index={split_index} out of range for {n_splits} splits"
        )
    shape = tuple(int(s) for s in mask.shape)
    if shape[0] == num_nodes:
        return mask[:, split_index]
    return mask[split_index]


def as_nodes_by_splits(mask, num_nodes: int):
    """Return a 2-D mask as ``[num_nodes, n_splits]`` (or 1-D unchanged)."""
    if mask is None or int(mask.ndim) != 2:
        return mask
    shape = tuple(int(s) for s in mask.shape)
    if shape[0] == num_nodes:
        return mask
    if shape[1] == num_nodes:
        return mask.t().contiguous()
    raise ValueError(
        f"Mask shape {shape} is incompatible with num_nodes={num_nodes}"
    )


def graphany_split_seed(
    load_seed: int,
    nc_split_index: Optional[int],
) -> int:
    """Sklearn ``random_state`` for GraphAny-generated splits.

    When ``nc_split_index`` is set (eval protocol), use it so splits 0..4 are
    GraphAny ``cfg.seed`` values. Otherwise keep ``load_seed`` (baselines that
    call ``DataSet.load(..., seed=42)``).
    """
    if nc_split_index is not None:
        return int(nc_split_index)
    return int(load_seed)


def predefined_split_index(nc_split_index: Optional[int]) -> int:
    """Column to use on official multi-split masks (None → first split)."""
    if nc_split_index is None:
        return 0
    return int(nc_split_index)


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print NC eval split indices for one dataset (space-separated)."
    )
    parser.add_argument("dataset", help="Registry key, e.g. TEXAS or PHOTO")
    parser.add_argument(
        "--mode",
        choices=("all", "first"),
        default="all",
        help="all = protocol splits; first = only split 0",
    )
    args = parser.parse_args(argv)
    indices = eval_nc_split_indices(args.dataset, mode=args.mode)
    print(" ".join(str(i) for i in indices))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
