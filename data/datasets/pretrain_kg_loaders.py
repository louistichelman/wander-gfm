"""Shared bundle loaders for transductive pretrain KG datasets."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from torch_geometric.datasets import FB15k_237, WordNet18RR

from ..graph_bundle import EdgeSetMode, GraphBundle
from .kg_eval.bundle_builder import load_three_split_transductive


def load_pretrain_kg_bundle(
    data_dir: str,
    name: str,
    raw_paths: List[Path],
    *,
    add_inverse_edges_kgs: bool,
    edge_set_mode: EdgeSetMode = "all_positives",
    delimiter: Optional[str] = "\t",
) -> GraphBundle:
    """Load a three-split transductive KG via ``build_transductive_bundle``."""
    return load_three_split_transductive(
        Path(data_dir),
        name,
        urls=[],
        filenames=[],
        add_inverse_edges_kgs=add_inverse_edges_kgs,
        edge_set_mode=edge_set_mode,
        delimiter=delimiter,
        raw_paths=raw_paths,
    )


def _ensure_pyg_kg_raw(
    data_dir: Path,
    name: str,
    pyg_cls,
    *,
    split: Optional[str] = None,
) -> List[Path]:
    """Trigger PyG download and return ``[train, valid, test]`` raw txt paths."""
    root = data_dir / name
    if split is not None:
        pyg_cls(root=str(root), split=split)
    else:
        pyg_cls(root=str(root))
    raw = root / "raw"
    return [raw / "train.txt", raw / "valid.txt", raw / "test.txt"]


def load_fb15k237_bundle(
    data_dir: str,
    *,
    add_inverse_edges_kgs: bool = True,
    edge_set_mode: EdgeSetMode = "all_positives",
) -> GraphBundle:
    paths = _ensure_pyg_kg_raw(Path(data_dir), "FB15k-237", FB15k_237, split="train")
    return load_pretrain_kg_bundle(
        data_dir,
        "FB15k-237",
        paths,
        add_inverse_edges_kgs=add_inverse_edges_kgs,
        edge_set_mode=edge_set_mode,
        delimiter="\t",
    )


def load_wn18rr_bundle(
    data_dir: str,
    *,
    add_inverse_edges_kgs: bool = True,
    edge_set_mode: EdgeSetMode = "all_positives",
) -> GraphBundle:
    paths = _ensure_pyg_kg_raw(Path(data_dir), "WN18RR", WordNet18RR)
    return load_pretrain_kg_bundle(
        data_dir,
        "WN18RR",
        paths,
        add_inverse_edges_kgs=add_inverse_edges_kgs,
        edge_set_mode=edge_set_mode,
        delimiter="\t",
    )


def ensure_codex_medium_raw(data_dir: Path) -> List[Path]:
    """Download CoDEx-Medium split files if missing."""
    import urllib.request

    data_path = data_dir / "CoDEx-Medium"
    data_path.mkdir(parents=True, exist_ok=True)
    base_url = "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-m"
    filenames = ("train.txt", "valid.txt", "test.txt")
    paths = []
    for filename in filenames:
        filepath = data_path / filename
        if not filepath.exists():
            url = f"{base_url}/{filename}"
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, str(filepath))
        paths.append(filepath)
    return paths


def load_codex_medium_bundle(
    data_dir: str,
    *,
    add_inverse_edges_kgs: bool = True,
    edge_set_mode: EdgeSetMode = "all_positives",
) -> GraphBundle:
    paths = ensure_codex_medium_raw(Path(data_dir))
    return load_pretrain_kg_bundle(
        data_dir,
        "CoDEx-Medium",
        paths,
        add_inverse_edges_kgs=add_inverse_edges_kgs,
        edge_set_mode=edge_set_mode,
        delimiter="\t",
    )
