"""WN18RR / WN18RR-Ind with PCA-reduced lemma+gloss MiniLM node features.

Synset IDs are 8-digit WordNet offsets. Text comes from the KG-BERT
``entity2text`` map (``lemma, gloss``), rendered as ``lemma: gloss``. The
offset-keyed cache is shared by the transductive graph and all GraIL inductive
versions so MiniLM is not an eval-time dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from ...graph_bundle import GraphBundle
from .bundle_builder import load_three_split_transductive
from .common import ensure_download, entity_vocab_from_triple_files, load_triple_file
from .conceptnet_feat import DEFAULT_PCA_DIM, pca_reduce
from .text_embed import DEFAULT_EMBED_MODEL, encode_texts, model_slug
from .inductive import load_grail_bundle

WN18RR_DISPLAY = "WN18RR-Feat"
WN18RR_IND_DISPLAY = "WN18RR-Ind-Feat"
WN18RR_RAW_NAME = "WN18RR"
WN18RR_IND_RAW_NAME = "WN18RR-Ind"
TEXT_RECIPE = "lemma_gloss"
_SPLIT_FILES = ("train.txt", "valid.txt", "test.txt")
ENTITY2TEXT_URL = (
    "https://raw.githubusercontent.com/yao8839836/kg-bert/master/data/WN18RR/"
    "entity2text.txt"
)


def cache_path(
    data_dir: Path,
    embed_model: str = DEFAULT_EMBED_MODEL,
    pca_dim: int = DEFAULT_PCA_DIM,
) -> Path:
    slug = model_slug(embed_model)
    fname = f"offset_features_{slug}_pca{int(pca_dim)}.pt"
    return data_dir / WN18RR_DISPLAY / "cache" / fname


def entity2text_path(data_dir: Path) -> Path:
    return data_dir / WN18RR_DISPLAY / "entity2text.txt"


def ensure_entity2text(data_dir: Path) -> Path:
    dest = entity2text_path(data_dir)
    ensure_download(ENTITY2TEXT_URL, dest)
    return dest


def parse_entity2text(path: Path) -> Dict[str, str]:
    """Parse KG-BERT ``offset<TAB>lemma, gloss`` lines into offset → raw text."""
    mapping: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            offset, sep, rest = line.partition("\t")
            if not sep:
                offset, sep, rest = line.partition(" ")
            if not sep:
                continue
            mapping[offset.strip()] = rest.strip()
    return mapping


def lemma_gloss_text(raw: str) -> str:
    """Turn KG-BERT ``lemma, gloss`` into MiniLM input ``lemma: gloss``."""
    raw = raw.strip()
    if ", " not in raw:
        return raw
    lemma, gloss = raw.split(", ", 1)
    lemma, gloss = lemma.strip(), gloss.strip()
    if gloss:
        return f"{lemma}: {gloss}"
    return lemma


def assemble_offset_texts(offset_to_raw: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """Preserve file order: parallel ``offsets`` and MiniLM strings."""
    offsets = list(offset_to_raw.keys())
    texts = [lemma_gloss_text(offset_to_raw[off]) for off in offsets]
    return offsets, texts


def save_offset_feature_cache(
    path: Path,
    *,
    model: str,
    pca_dim: int,
    offsets: List[str],
    embeddings: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model,
            "pca_dim": int(pca_dim),
            "text_recipe": TEXT_RECIPE,
            "offsets": list(offsets),
            "embeddings": embeddings,
        },
        path,
    )


def load_offset_feature_cache(
    path: Path,
    *,
    expected_model: str = DEFAULT_EMBED_MODEL,
    expected_pca_dim: int = DEFAULT_PCA_DIM,
) -> Tuple[Dict[str, int], torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing WN18RR node feature cache: {path}\n"
            "Run precompute first, e.g.:\n"
            "  python scripts/data/precompute_wn18rr_node_features.py "
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
    offsets = payload["offsets"]
    embeddings = payload["embeddings"]
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)
    embeddings = embeddings.float()
    if len(offsets) != embeddings.shape[0]:
        raise ValueError(
            f"Cache offset count ({len(offsets)}) != embedding rows "
            f"({embeddings.shape[0]}) in {path}"
        )
    index = {str(off): i for i, off in enumerate(offsets)}
    return index, embeddings


def features_for_vocab(
    vocab: Dict[str, int],
    offset_index: Dict[str, int],
    embeddings: torch.Tensor,
) -> torch.Tensor:
    """Assemble a ``[num_nodes, pca_dim]`` matrix aligned to integer entity ids."""
    if not vocab:
        raise ValueError("Empty entity vocabulary")
    n = max(vocab.values()) + 1
    x = torch.empty(n, embeddings.shape[1], dtype=torch.float32)
    missing: List[str] = []
    for tok, idx in vocab.items():
        row = offset_index.get(tok)
        if row is None:
            missing.append(tok)
            continue
        x[idx] = embeddings[row]
    if missing:
        sample = ", ".join(missing[:5])
        raise KeyError(
            f"{len(missing)} WN offsets missing from lemma+gloss cache "
            f"(e.g. {sample}). Re-run precompute_wn18rr_node_features.py."
        )
    return x


def wn18rr_raw_paths(data_dir: Path) -> List[Path]:
    raw = data_dir / WN18RR_RAW_NAME / "raw"
    paths = [raw / fn for fn in _SPLIT_FILES]
    if all(p.exists() for p in paths):
        return paths
    from torch_geometric.datasets import WordNet18RR

    from ..pretrain_kg_loaders import _ensure_pyg_kg_raw

    return _ensure_pyg_kg_raw(data_dir, WN18RR_RAW_NAME, WordNet18RR)


def grail_entity_vocabs(raw: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Rebuild train/test entity vocabs in the same order as ``load_grail_bundle``."""
    _, _, _, train_ev, train_rv = load_triple_file(raw / "train.txt")
    _, _, _, train_ev, train_rv = load_triple_file(
        raw / "valid.txt", inv_entity_vocab=train_ev, inv_rel_vocab=train_rv
    )
    _, _, _, test_ev, test_rv = load_triple_file(raw / "train_ind.txt")
    _, _, _, test_ev, test_rv = load_triple_file(
        raw / "valid_ind.txt", inv_entity_vocab=test_ev, inv_rel_vocab=test_rv
    )
    _, _, _, test_ev, test_rv = load_triple_file(
        raw / "test_ind.txt", inv_entity_vocab=test_ev, inv_rel_vocab=test_rv
    )
    del train_rv, test_rv
    return train_ev, test_ev


def _attach_transductive_features(
    bundle: GraphBundle,
    vocab: Dict[str, int],
    offset_index: Dict[str, int],
    embeddings: torch.Tensor,
    display: str,
) -> GraphBundle:
    x = features_for_vocab(vocab, offset_index, embeddings)
    if x.shape[0] != bundle.train.num_nodes:
        raise ValueError(
            f"Feature matrix rows ({x.shape[0]}) != num_nodes "
            f"({bundle.train.num_nodes}) for {display}"
        )
    bundle.train.x = x
    bundle.has_node_features = True
    return bundle


def load_wn18rr_feat_bundle(
    data_dir: str,
    add_inverse_edges_kgs: bool,
    *,
    embed_model: str = DEFAULT_EMBED_MODEL,
    pca_dim: int = DEFAULT_PCA_DIM,
) -> GraphBundle:
    """Load transductive WN18RR triples and attach cached lemma+gloss MiniLM features."""
    root = Path(data_dir)
    raw_paths = wn18rr_raw_paths(root)
    _, inv_entity_vocab = entity_vocab_from_triple_files(
        raw_paths, delimiter="\t"
    )
    bundle = load_three_split_transductive(
        root,
        WN18RR_DISPLAY,
        [],
        list(_SPLIT_FILES),
        add_inverse_edges_kgs,
        edge_set_mode="all_positives",
        delimiter="\t",
        raw_paths=raw_paths,
    )
    offset_index, embeddings = load_offset_feature_cache(
        cache_path(root, embed_model, pca_dim),
        expected_model=embed_model,
        expected_pca_dim=pca_dim,
    )
    return _attach_transductive_features(
        bundle, inv_entity_vocab, offset_index, embeddings, WN18RR_DISPLAY
    )


def load_wn18rr_ind_feat_bundle(
    data_dir: str,
    version: str,
    add_inverse_edges_kgs: bool,
    *,
    embed_model: str = DEFAULT_EMBED_MODEL,
    pca_dim: int = DEFAULT_PCA_DIM,
) -> GraphBundle:
    """Load a GraIL WN18RR-Ind version and attach lemma+gloss MiniLM on every split."""
    root = Path(data_dir)
    bundle = load_grail_bundle(
        root,
        WN18RR_IND_RAW_NAME,
        "WN18RR",
        version,
        add_inverse_edges_kgs,
    )
    bundle.name = WN18RR_IND_DISPLAY
    raw = root / WN18RR_IND_RAW_NAME / version / "raw"
    train_ev, test_ev = grail_entity_vocabs(raw)
    offset_index, embeddings = load_offset_feature_cache(
        cache_path(root, embed_model, pca_dim),
        expected_model=embed_model,
        expected_pca_dim=pca_dim,
    )
    train_x = features_for_vocab(train_ev, offset_index, embeddings)
    test_x = features_for_vocab(test_ev, offset_index, embeddings)
    if train_x.shape[0] != bundle.train.num_nodes:
        raise ValueError(
            f"Train feature rows ({train_x.shape[0]}) != num_nodes "
            f"({bundle.train.num_nodes}) for {WN18RR_IND_DISPLAY}:{version}"
        )
    if bundle.test is None:
        raise ValueError(f"{WN18RR_IND_DISPLAY}:{version} has no test graph")
    if test_x.shape[0] != bundle.test.num_nodes:
        raise ValueError(
            f"Test feature rows ({test_x.shape[0]}) != num_nodes "
            f"({bundle.test.num_nodes}) for {WN18RR_IND_DISPLAY}:{version}"
        )
    bundle.train.x = train_x
    if bundle.val is not None:
        if int(bundle.val.num_nodes) != int(bundle.train.num_nodes):
            raise ValueError(
                f"Val num_nodes ({bundle.val.num_nodes}) != train "
                f"({bundle.train.num_nodes}) for {WN18RR_IND_DISPLAY}:{version}"
            )
        bundle.val.x = train_x.clone()
    bundle.test.x = test_x
    bundle.has_node_features = True
    return bundle


def precompute_node_features(
    data_dir: str,
    *,
    embed_model: str = DEFAULT_EMBED_MODEL,
    pca_dim: int = DEFAULT_PCA_DIM,
) -> Path:
    """Embed every WN18RR offset's lemma+gloss, PCA-reduce, and write the cache."""
    root = Path(data_dir)
    mapping = parse_entity2text(ensure_entity2text(root))
    if not mapping:
        raise RuntimeError(f"Empty entity2text map at {entity2text_path(root)}")
    offsets, texts = assemble_offset_texts(mapping)
    embeddings = pca_reduce(
        encode_texts(texts, embed_model, batch_size=256), pca_dim
    )
    out = cache_path(root, embed_model, pca_dim)
    save_offset_feature_cache(
        out,
        model=embed_model,
        pca_dim=pca_dim,
        offsets=offsets,
        embeddings=embeddings,
    )
    return out
