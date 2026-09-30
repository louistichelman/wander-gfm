"""Download/extract helper for AnyGraph link-prediction datasets.

The AnyGraph zero-shot benchmarks ship as a single (~5 GB) archive on
HuggingFace (``hkuds/AnyGraph_datasets``). The archive extracts to a
``zero-shot datasets/<name>/`` folder layout, each containing
``trn_mat.pkl`` / ``val_mat.pkl`` / ``tst_mat.pkl`` and optionally
``feats.pkl`` (see ``AnyGraph/data_handler.py``). We download + extract once
into ``--data_dir`` and reuse the extracted folders thereafter.
"""

from __future__ import annotations

import pickle
import zipfile
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from ..kg_eval.common import ensure_download

# The repo exposes exactly one LFS file: "zero-shot datasets.zip".
ANYGRAPH_ZIP_URL = (
    "https://huggingface.co/datasets/hkuds/AnyGraph_datasets/"
    "resolve/main/zero-shot%20datasets.zip"
)
# Top-level folder inside the archive.
ANYGRAPH_INNER_DIR = "zero-shot datasets"


def anygraph_trn_matrix_shape(data_dir: str | Path, folder: str) -> tuple[int, int]:
    """Return ``(n_rows, n_cols)`` of the raw AnyGraph training adjacency matrix."""
    ds_dir = ensure_anygraph_raw(Path(data_dir), folder)
    with open(ds_dir / "trn_mat.pkl", "rb") as fh:
        mat = pickle.load(fh)
    if sp.issparse(mat):
        return int(mat.shape[0]), int(mat.shape[1])
    arr = np.asarray(mat)
    return int(arr.shape[0]), int(arr.shape[1])


def anygraph_adjacency_is_rectangular(data_dir: str | Path, folder: str) -> bool:
    """True when ``trn_mat`` is user x item (non-square), i.e. bipartite layout."""
    n_rows, n_cols = anygraph_trn_matrix_shape(data_dir, folder)
    return n_rows != n_cols


def ensure_anygraph_raw(data_dir: Path, folder: str) -> Path:
    """Ensure the AnyGraph dataset folder ``folder`` exists locally.

    Downloads and extracts the shared archive once, then returns the path to
    ``<data_dir>/AnyGraph/zero-shot datasets/<folder>/``.
    """
    base = Path(data_dir) / "AnyGraph"
    extracted_root = base / ANYGRAPH_INNER_DIR
    ds_dir = extracted_root / folder
    if (ds_dir / "trn_mat.pkl").exists():
        return ds_dir

    base.mkdir(parents=True, exist_ok=True)
    zip_path = base / "zero-shot_datasets.zip"
    ensure_download(ANYGRAPH_ZIP_URL, zip_path, validate_zip=True, retries=3)

    if not (ds_dir / "trn_mat.pkl").exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(base)

    if not (ds_dir / "trn_mat.pkl").exists():
        raise FileNotFoundError(
            f"AnyGraph dataset '{folder}' not found after extraction at {ds_dir}. "
            f"Expected '{ANYGRAPH_INNER_DIR}/{folder}/trn_mat.pkl' inside the archive."
        )
    return ds_dir
