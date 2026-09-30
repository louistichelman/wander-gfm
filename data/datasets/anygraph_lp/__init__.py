"""AnyGraph link-prediction dataset loaders (single-relation transductive KGs)."""

from .register import (
    register_anygraph_datasets,
    build_anygraph_registry,
)

__all__ = [
    "register_anygraph_datasets",
    "build_anygraph_registry",
]
