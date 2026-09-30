"""Shared MiniLM helpers for KG node-feature caches (ConceptNet / WN18RR)."""

from __future__ import annotations

import re
from typing import List

import torch

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def model_slug(model_name: str) -> str:
    """Filesystem-safe slug for embedding model id."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name)


def encode_texts(
    texts: List[str],
    model_name: str = DEFAULT_EMBED_MODEL,
    batch_size: int = 64,
) -> torch.Tensor:
    """Encode entity texts with SentenceTransformer (precompute only)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return torch.from_numpy(vectors).float()
