"""Data loading module for Wander.

This module provides dataset definitions and transform utilities for loading
and preprocessing graph data for link prediction and node classification tasks.

"""

from .dataset import DataSet, get_datasetargs, GraphBundle, bundle_to_legacy_data

__all__ = [
    "DataSet",
    "get_datasetargs",
    "GraphBundle",
    "bundle_to_legacy_data",
]
