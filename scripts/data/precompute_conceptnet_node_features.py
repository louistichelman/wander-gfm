"""Precompute PCA-reduced MiniLM node features for CONCEPTNET100K_FEAT.

Usage (from the repo root):
    python scripts/data/precompute_conceptnet_node_features.py --data_dir raw_data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.datasets.kg_eval.text_embed import DEFAULT_EMBED_MODEL  # noqa: E402
from data.datasets.kg_eval.conceptnet_feat import (  # noqa: E402
    DEFAULT_PCA_DIM,
    precompute_node_features,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute ConceptNet raw-string MiniLM node feature caches."
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
    print("=== Precomputing ConceptNet-100k node features ===")
    out = precompute_node_features(
        args.data_dir,
        embed_model=args.model,
        pca_dim=args.pca_dim,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
