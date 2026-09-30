"""ConceptNet-100k with PCA-reduced raw-string MiniLM node features."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.decomposition import PCA

from ...graph_bundle import GraphBundle
from .bundle_builder import load_three_split_transductive
from .common import ensure_download, entity_vocab_from_triple_files
from .text_embed import DEFAULT_EMBED_MODEL, encode_texts, model_slug

CONCEPTNET_DISPLAY = "ConceptNet100k-Feat"
CONCEPTNET_RAW_NAME = "ConceptNet100k"
DEFAULT_PCA_DIM = 32
TEXT_RECIPE = "raw_entity_string"
_SPLIT_FILES = ("train.txt", "valid.txt", "test.txt")
CONCEPTNET_URLS = [
    "https://raw.githubusercontent.com/guojiapub/BiQUE/master/src_data/conceptnet-100k/train",
    "https://raw.githubusercontent.com/guojiapub/BiQUE/master/src_data/conceptnet-100k/valid",
    "https://raw.githubusercontent.com/guojiapub/BiQUE/master/src_data/conceptnet-100k/test",
]


def cache_path(
    data_dir: Path,
    embed_model: str = DEFAULT_EMBED_MODEL,
    pca_dim: int = DEFAULT_PCA_DIM,
) -> Path:
    slug = model_slug(embed_model)
    fname = f"node_features_{slug}_pca{int(pca_dim)}.pt"
    return data_dir / CONCEPTNET_DISPLAY / "cache" / fname


def conceptnet_raw_paths(data_dir: Path) -> List[Path]:
    raw = data_dir / CONCEPTNET_RAW_NAME / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    paths = []
    for url, fn in zip(CONCEPTNET_URLS, _SPLIT_FILES):
        dest = raw / fn
        ensure_download(url, dest)
        paths.append(dest)
    return paths


def assemble_entity_texts(inv_entity_vocab: Dict[str, int]) -> List[str]:
    """One raw entity string per integer id (ConceptNet concepts are already text)."""
    id_to_token = {idx: tok for tok, idx in inv_entity_vocab.items()}
    n = len(inv_entity_vocab)
    return [id_to_token.get(i, f"entity_{i}") for i in range(n)]


def pca_reduce(embeddings: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Project MiniLM vectors to ``target_dim`` (no-op if already smaller)."""
    if target_dim <= 0 or embeddings.shape[1] <= target_dim:
        return embeddings.float()
    k = min(int(target_dim), embeddings.shape[0], embeddings.shape[1])
    reduced = PCA(n_components=k, random_state=0).fit_transform(
        embeddings.detach().cpu().numpy()
    )
    return torch.from_numpy(np.asarray(reduced, dtype=np.float32))


def save_node_feature_cache(
    path: Path,
    *,
    model: str,
    pca_dim: int,
    entity_vocab: Dict[str, int],
    embeddings: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model,
            "pca_dim": int(pca_dim),
            "text_recipe": TEXT_RECIPE,
            "entity_vocab": entity_vocab,
            "embeddings": embeddings,
        },
        path,
    )


def load_node_feature_cache(
    path: Path,
    *,
    expected_model: str = DEFAULT_EMBED_MODEL,
    expected_pca_dim: int = DEFAULT_PCA_DIM,
    expected_vocab: Optional[Dict[str, int]] = None,
) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing ConceptNet node feature cache: {path}\n"
            "Run precompute first, e.g.:\n"
            "  python scripts/data/precompute_conceptnet_node_features.py "
            f"--data_dir {path.parents[2]}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model") != expected_model:
        raise ValueError(
            f"Cache model mismatch: expected {expected_model!r}, "
            f"got {payload.get('model')!r} in {path}"
        )
    if int(payload.get("pca_dim", -1)) != int(expected_pca_dim):
        raise ValueError(
            f"Cache PCA dim mismatch: expected {expected_pca_dim}, "
            f"got {payload.get('pca_dim')!r} in {path}"
        )
    if expected_vocab is not None and payload.get("entity_vocab") != expected_vocab:
        raise ValueError(
            f"Entity vocabulary mismatch for cache {path}. "
            "Re-run precompute after triple files changed."
        )
    embeddings = payload["embeddings"]
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)
    return embeddings.float()


def load_conceptnet_feat_bundle(
    data_dir: str,
    add_inverse_edges_kgs: bool,
    *,
    embed_model: str = DEFAULT_EMBED_MODEL,
    pca_dim: int = DEFAULT_PCA_DIM,
) -> GraphBundle:
    """Load ConceptNet-100k triples and attach cached raw-string MiniLM features."""
    root = Path(data_dir)
    raw_paths = conceptnet_raw_paths(root)
    _, inv_entity_vocab = entity_vocab_from_triple_files(
        raw_paths, delimiter="\t"
    )
    bundle = load_three_split_transductive(
        root,
        CONCEPTNET_DISPLAY,
        CONCEPTNET_URLS,
        list(_SPLIT_FILES),
        add_inverse_edges_kgs,
        delimiter="\t",
        raw_paths=raw_paths,
    )
    x = load_node_feature_cache(
        cache_path(root, embed_model, pca_dim),
        expected_model=embed_model,
        expected_pca_dim=pca_dim,
        expected_vocab=inv_entity_vocab,
    )
    if x.shape[0] != bundle.train.num_nodes:
        raise ValueError(
            f"Feature matrix rows ({x.shape[0]}) != num_nodes "
            f"({bundle.train.num_nodes}) for {CONCEPTNET_DISPLAY}"
        )
    bundle.train.x = x
    bundle.has_node_features = True
    return bundle


def precompute_node_features(
    data_dir: str,
    *,
    embed_model: str = DEFAULT_EMBED_MODEL,
    pca_dim: int = DEFAULT_PCA_DIM,
) -> Path:
    """Embed ConceptNet entity strings, PCA-reduce, and write the cache file."""
    root = Path(data_dir)
    raw_paths = conceptnet_raw_paths(root)
    _, inv_entity_vocab = entity_vocab_from_triple_files(
        raw_paths, delimiter="\t"
    )
    texts = assemble_entity_texts(inv_entity_vocab)
    embeddings = pca_reduce(
        encode_texts(texts, embed_model, batch_size=256), pca_dim
    )
    out = cache_path(root, embed_model, pca_dim)
    save_node_feature_cache(
        out,
        model=embed_model,
        pca_dim=pca_dim,
        entity_vocab=inv_entity_vocab,
        embeddings=embeddings,
    )
    return out
