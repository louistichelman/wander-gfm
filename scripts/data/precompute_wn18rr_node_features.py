"""Precompute PCA-reduced MiniLM node features for WN18RR_FEAT / WN18RR_IND_FEAT.

Usage (from the repo root):
    python scripts/data/precompute_wn18rr_node_features.py --data_dir raw_data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.datasets.kg_eval.text_embed import DEFAULT_EMBED_MODEL  # noqa: E402
from data.datasets.kg_eval.wn_feat import (  # noqa: E402
    DEFAULT_PCA_DIM,
    precompute_node_features,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute WN18RR lemma+gloss MiniLM node feature caches."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="raw_data",
        help="Root dataset directory (same as --data_dir in main.py).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_EMBED_MODEL,
        help=f"SentenceTransformer model id (default: {DEFAULT_EMBED_MODEL}).",
    )
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=DEFAULT_PCA_DIM,
        help=f"PCA target dim written into the cache (default: {DEFAULT_PCA_DIM}).",
    )
    args = parser.parse_args()
    print("=== Precomputing WN18RR lemma+gloss node features ===")
    out = precompute_node_features(
        args.data_dir,
        embed_model=args.model,
        pca_dim=args.pca_dim,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
