"""Dataset definitions and loading utilities for Wander."""

from typing import Callable, Optional, Set, Tuple
import inspect

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected, to_undirected
from sklearn.model_selection import train_test_split
import numpy as np

from .graph_bundle import (
    GraphBundle,
    EdgeSetMode,
    build_edge_set as build_edge_set_from_graph,
    wrap_transductive_data,
    bundle_to_legacy_data,
    is_link_prediction as is_link_prediction_data,
)
from .transforms import (
    RemoveSelfLoops,
    NormalizeFeatures,
    PCAFeatures,
    DropConstantTrainFeatures,
    apply_graphland_per_type_transforms,
)
from .nc_splits import (
    graphany_split_seed,
    n_predefined_splits_from_masks,
    predefined_split_index,
    select_node_split_mask,
)

def _max_finite_label(y: torch.Tensor) -> torch.Tensor:
    """Reduce max over finite entries only (avoids NaN propagation; works without ``torch.nanmax``)."""
    yf = y[torch.isfinite(y)]
    if yf.numel() == 0:
        raise ValueError("No finite labels in y")
    return yf.max()


def _num_classes_from_int_labels(y: torch.Tensor) -> int:
    """Infer class count from 1D per-node labels.

    GraphLand (and similar) leave ``float('nan')`` on unlabeled nodes; :meth:`~torch.Tensor.max`
    then propagates NaN, so we take the max over finite values only.
    """
    if y.ndim != 1:
        raise ValueError("Expected 1D label tensor for class count")
    if y.dtype.is_floating_point and torch.isnan(y).any():
        return int(_max_finite_label(y).item()) + 1
    return int(y.max().item()) + 1


def _nan_node_labels_to_dummy(y: torch.Tensor, dummy: float = 0.0) -> torch.Tensor:
    """Replace NaN node labels so downstream ops (e.g. ``one_hot``) are valid.

    Unlabeled nodes are typically excluded from train/val/test masks; dummy values
    are not used by the objective when masks are applied.
    """
    if y.dtype.is_floating_point and torch.isnan(y).any():
        return torch.nan_to_num(y, nan=dummy)
    return y


def apply_dummy_features(data: Data) -> Data:
    """Replace node features with a single constant-zero column."""
    data.x = torch.zeros(data.num_nodes, 1, dtype=torch.float32)
    return data


def effective_pca_target_dim(
    base_dim: int,
    num_nodes: int,
    adaptive_pca_node_threshold: int,
) -> int:
    """Halve ``base_dim`` when ``num_nodes`` exceeds ``adaptive_pca_node_threshold``.

    When ``adaptive_pca_node_threshold`` is 0 (default), ``base_dim`` is returned unchanged.
    """
    if adaptive_pca_node_threshold <= 0 or num_nodes <= adaptive_pca_node_threshold:
        return base_dim
    return max(1, base_dim // 2)


def apply_feature_preprocessing(
    data: Data,
    *,
    pca_target_dim: int,
    has_node_features: bool = True,
    dummy_features: bool = False,
    ignore_features: bool = False,
    ignore_edge_types: bool = False,
    row_wise_norming: bool = False,
    row_norm_mode: str = "l2",
    ensure_edge_type: bool = True,
    pca_before_normalization: bool = True,
    graphland_different_transform: bool = False,
    graphland_categorical_as_ordinals: bool = False,
    drop_constant_train_features: bool = False,
    final_inductive_zscore: bool = False,
) -> Data:
    """Shared node-feature pipeline for real-world and synthetic graphs.

    Order (default): remove self-loops, PCA, normalization,
    then optional dummy/ignore overrides.

    When ``pca_before_normalization=False``, normalization runs before PCA
    (GraphPFN eval order). When ``graphland_different_transform=True`` and
    column-type masks are present, per-type quantile transforms replace column
    z-score (GraphPFN GraphLand eval path: quantiles → PCA → drop constant).

    Column z-score uses statistics over all nodes (transductive), even when
    ``data.train_mask`` is set.

    When ``final_inductive_zscore=True``, a per-column z-score is appended as
    the very last step (after PCA / per-type transforms / drop-constant) using
    train-node statistics only, clipped to ``+-100``. This mimics the input
    normalization LimiX/GraphPFN applies inside the model (``normalize_x=True``,
    ``normalize_on_train_only=True``) so the model sees GraphPFN-equivalent
    inputs on all paths. Falls back to all-node statistics when ``train_mask``
    is missing or invalid.

    """
    data = RemoveSelfLoops()(data)

    train_mask = getattr(data, "train_mask", None)
    zscore_mask = train_mask

    if (
        has_node_features
        and getattr(data, "x", None) is not None
        and not dummy_features
        and not ignore_features
    ):
        use_graphland_quantiles = (
            graphland_different_transform and hasattr(data, "x_numerical_mask")
        )
        if use_graphland_quantiles:
            data = apply_graphland_per_type_transforms(
                data,
                train_mask=train_mask,
                categorical_as_ordinals=graphland_categorical_as_ordinals,
            )

        if use_graphland_quantiles:
            data = PCAFeatures(target_dim=pca_target_dim)(data)
        else:
            if row_wise_norming:
                if row_norm_mode not in ("l1", "l2"):
                    raise ValueError(
                        f"row_norm_mode must be 'l1' or 'l2', got {row_norm_mode!r}"
                    )
                mode = f"row_{row_norm_mode}"
            else:
                mode = "col_zscore"
            norm = NormalizeFeatures(mode=mode, train_mask=None)

            if pca_before_normalization:
                data = PCAFeatures(target_dim=pca_target_dim)(data)
                data = norm(data)
            else:
                data = norm(data)
                data = PCAFeatures(target_dim=pca_target_dim)(data)

        if drop_constant_train_features:
            data = DropConstantTrainFeatures(train_mask=train_mask)(data)

        if final_inductive_zscore:
            data = NormalizeFeatures(
                mode="col_zscore",
                train_mask=zscore_mask,
                clip_value=100.0,
            )(data)

    if dummy_features:
        data = apply_dummy_features(data)
    elif ignore_features:
        data.x = None

    if ensure_edge_type and (
        not hasattr(data, "edge_type") or data.edge_type is None
    ):
        data.edge_type = torch.zeros(data.edge_index.size(1), dtype=torch.long)
        data.num_relations = 1

    if ignore_edge_types:
        n_edges = int(data.edge_index.size(1))
        data.edge_type = torch.zeros(
            n_edges, dtype=torch.long, device=data.edge_index.device
        )
        data.num_relations = 1

    return data


def _ensure_undirected_homogeneous(data: Data) -> Data:
    """If ``edge_index`` is directed, replace it with an undirected (bidirectional) version.

    When ``edge_type`` is present with one entry per edge (e.g. a single relation), duplicate
    types for reverse edges.
    """
    ei = data.edge_index
    if ei.numel() == 0:
        return data
    num_nodes = data.num_nodes
    if num_nodes is None:
        num_nodes = int(ei.max().item()) + 1
        data.num_nodes = num_nodes
    if is_undirected(ei, num_nodes=num_nodes):
        return data
    et = data.edge_type if hasattr(data, "edge_type") and data.edge_type is not None else None
    if et is not None and et.numel() == ei.size(1):
        out = to_undirected(ei, et, num_nodes=num_nodes)
        data.edge_index, data.edge_type = out[0], out[1]
    else:
        data.edge_index = to_undirected(ei, num_nodes=num_nodes)
    return data


def build_edge_set(data: Data, mode: EdgeSetMode = "all_positives") -> Set[Tuple[int, int, int]]:
    """Build a set of (head, relation, tail) triples for negative-sampling filtering."""
    return build_edge_set_from_graph(data, mode)


def get_datasetargs(
    dataset_name: str,
    *,
    dataset_version: Optional[str] = None,
) -> dict:
    """Get dataset configuration by name.
    
    Args:
        dataset_name: Name of the dataset (case-insensitive).
        
    Returns:
        Dictionary containing dataset configuration with keys:
            - name: Canonical dataset name
            - has_node_features: Whether nodes have feature vectors
            - has_node_labels: Whether nodes have categorical labels
            - has_predefined_node_split: Whether a standard train/val/test node split exists
            - has_predefined_edge_split: Whether a standard train/val/test edge split exists
            - is_knowledge_graph: Whether this is a multi-relational KG (no forced undirecting).
            - load_base_dataset: Function to load the raw dataset
            
    Raises:
        KeyError: If dataset_name is not found in the registry.
    """
    from .datasets import DATASET_REGISTRY
    
    raw_name = dataset_name
    version = dataset_version
    if ":" in dataset_name and version is None:
        dataset_key, version = dataset_name.upper().split(":", 1)
    else:
        dataset_key = dataset_name.upper()
    
    if dataset_key not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise KeyError(f"Unknown dataset: {dataset_name}. Available: {available}")
    
    module = DATASET_REGISTRY[dataset_key]
    load_fn = getattr(module, "load_base_dataset", None)
    load_bundle_fn = getattr(module, "load_graph_bundle", None)
    if load_fn is None and load_bundle_fn is None:
        raise KeyError(f"Dataset {dataset_key} has no loader.")
    is_kg = getattr(module, "IS_KNOWLEDGE_GRAPH", False)
    supported = getattr(module, "SUPPORTED_VERSIONS", None)
    if version is not None and supported is not None and version.lower() not in [
        v.lower() for v in supported
    ]:
        raise KeyError(
            f"Dataset {dataset_key} does not support version '{version}'. "
            f"Available: {supported}"
        )
    return {
        "name": module.NAME,
        "registry_key": dataset_key,
        "has_node_features": module.HAS_NODE_FEATURES,
        "has_node_labels": module.HAS_NODE_LABELS,
        "has_predefined_node_split": module.HAS_PREDEFINED_NODE_SPLIT,
        "has_predefined_edge_split": module.HAS_PREDEFINED_EDGE_SPLIT,
        "is_knowledge_graph": is_kg,
        "is_inductive": getattr(module, "IS_INDUCTIVE", False),
        "inductive_filter_mode": getattr(module, "INDUCTIVE_FILTER_MODE", "transductive"),
        "edge_set_mode": getattr(module, "EDGE_SET_MODE", "train_only" if is_kg else "all_positives"),
        "preferred_link_pred_eval": getattr(module, "PREFERRED_LINK_PRED_EVAL", None),
        "preferred_evaluate_head_predictions": getattr(
            module, "PREFERRED_EVALUATE_HEAD_PREDICTIONS", None
        ),
        "preferred_eval_split": getattr(module, "PREFERRED_EVAL_SPLIT", None),
        "num_predefined_node_splits": getattr(
            module, "NUM_PREDEFINED_NODE_SPLITS", None
        ),
        "supported_versions": supported,
        "dataset_version": version.lower() if version else None,
        "load_base_dataset": load_fn,
        "load_graph_bundle": load_bundle_fn,
    }


class DataSet:
    """Dataset class with metadata and loading functionality.
    
    Each dataset has associated metadata describing its properties:
    - has_node_features: Whether nodes have feature vectors
    - has_node_labels: Whether nodes have categorical labels
    - has_predefined_node_split: Whether a standard train/val/test node split exists
    - has_predefined_edge_split: Whether a standard train/val/test edge split exists

    ``load()`` uses ``load_graph_bundle`` when defined (KG / link prediction); otherwise it
    runs the node-classification pipeline via ``load_base_dataset``.
    
    Example:
        >>> from data import DataSet, get_datasetargs
        >>> dataset_args = get_datasetargs("CORA")
        >>> dataset = DataSet(dataset_args)
        >>> data = dataset.load(data_dir="./data", seed=42)
        >>> data.edge_index.shape
        torch.Size([2, ...])
    """
    
    def __init__(
        self,
        dataset_args: dict,
        ignore_features: bool = False,
        ignore_edge_types: bool = False,
        dummy_features: bool = False,
        pca_target_dim: int = 2048,
        adaptive_pca_node_threshold: int = 0,
        always_generate_split: bool = False,
        row_wise_norming: bool = False,
        row_norm_mode: str = "l2",
        add_inverse_edges_kgs: bool = True,
        pca_before_normalization: bool = True,
        graphland_different_transform: bool = False,
        graphland_categorical_as_ordinals: bool = False,
        drop_constant_train_features: bool = False,
        final_inductive_zscore: bool = False,
        drop_feature_indices: Optional[list[int]] = None,
        nc_split_index: Optional[int] = None,
    ):
        """Initialize the dataset with configuration.
        
        Args:
            dataset_args: Dictionary from get_datasetargs() containing:
                - name: Canonical dataset name
                - has_node_features: Whether nodes have feature vectors
                - has_node_labels: Whether nodes have categorical labels
                - has_predefined_node_split: Whether a standard train/val/test node split exists
                - has_predefined_edge_split: Whether a standard train/val/test edge split exists
                - load_base_dataset: Function to load the raw dataset (node classification)
                - load_graph_bundle: Optional loader for KG / link-prediction datasets
            add_inverse_edges_kgs: Passed through to ``load_graph_bundle`` loaders that support it.
            always_generate_split: If True, always generate a new node split even
                when the dataset provides a predefined one.
            nc_split_index: Which official node split (2-D mask column) or
                GraphAny sklearn seed to use. ``None`` keeps legacy behaviour
                (first official split, or ``load(..., seed=)`` for GraphAny).
            pca_target_dim: PCA target dimension for real-world datasets (node cls and LP).
            adaptive_pca_node_threshold: When > 0, halve ``pca_target_dim`` for graphs with
                more nodes than this threshold (0 disables).
            row_wise_norming: If True, use per-row normalization after PCA (see
                ``row_norm_mode``); otherwise transductive column z-score (all nodes).
            row_norm_mode: When ``row_wise_norming`` is True, either ``"l1"`` or
                ``"l2"`` (default, unit L2 norm).
            drop_feature_indices: Optional 0-based raw feature column indices to drop,
                before PCA / normalization (ablation helper).
        """
        self.name: str = dataset_args["name"]
        self.has_node_features: bool = dataset_args["has_node_features"]
        self.pca_target_dim: int = pca_target_dim
        self.adaptive_pca_node_threshold: int = adaptive_pca_node_threshold
        self.ignore_features: bool = ignore_features
        self.ignore_edge_types: bool = ignore_edge_types
        self.dummy_features: bool = dummy_features
        self.has_node_labels: bool = dataset_args["has_node_labels"]
        self.has_predefined_node_split: bool = dataset_args["has_predefined_node_split"]
        self.has_predefined_edge_split: bool = dataset_args["has_predefined_edge_split"]
        self.is_knowledge_graph: bool = dataset_args.get("is_knowledge_graph", False)
        self.is_inductive: bool = dataset_args.get("is_inductive", False)
        self.inductive_filter_mode: str = dataset_args.get("inductive_filter_mode", "transductive")
        self.edge_set_mode: EdgeSetMode = dataset_args.get("edge_set_mode", "all_positives")
        self.preferred_link_pred_eval: Optional[str] = dataset_args.get("preferred_link_pred_eval")
        self.preferred_evaluate_head_predictions: Optional[bool] = dataset_args.get(
            "preferred_evaluate_head_predictions"
        )
        self.preferred_eval_split: Optional[str] = dataset_args.get("preferred_eval_split")
        self.dataset_version: Optional[str] = dataset_args.get("dataset_version")
        self.add_inverse_edges_kgs: bool = add_inverse_edges_kgs
        self._load_base_dataset_fn: Callable[..., Data] = dataset_args["load_base_dataset"]
        self._load_graph_bundle_fn = dataset_args.get("load_graph_bundle")
        self.always_generate_split: bool = always_generate_split
        self.nc_split_index: Optional[int] = (
            None if nc_split_index is None else int(nc_split_index)
        )
        self.row_wise_norming: bool = row_wise_norming
        self.row_norm_mode: str = row_norm_mode
        self.pca_before_normalization: bool = pca_before_normalization
        self.graphland_different_transform: bool = graphland_different_transform
        self.graphland_categorical_as_ordinals: bool = graphland_categorical_as_ordinals
        self.drop_constant_train_features: bool = drop_constant_train_features
        self.final_inductive_zscore: bool = final_inductive_zscore
        self.drop_feature_indices: Optional[list[int]] = (
            list(drop_feature_indices) if drop_feature_indices else None
        )
        self.bundle: Optional[GraphBundle] = None
        self.data: Optional[Data] = None  # legacy alias: train split
    
    def load(
        self,
        data_dir: str,
        seed: int = 42,
    ) -> GraphBundle:
        """Load and preprocess the dataset as a GraphBundle."""
        torch.manual_seed(seed)

        if self._load_graph_bundle_fn is not None:
            bundle_kwargs = {
                "seed": seed,
                "add_inverse_edges_kgs": self.add_inverse_edges_kgs,
                "dataset_version": self.dataset_version,
            }
            bundle = self._load_graph_bundle_fn(data_dir, **bundle_kwargs)
            bundle = self._postprocess_bundle(bundle, seed)
            self._stamp_eval_pref(bundle)
            self.bundle = bundle
            self.data = bundle.train
            return bundle

        data = self._load_base_dataset_fn(
            data_dir,
            **self._graphland_load_kwargs(),
        )
        data = _ensure_undirected_homogeneous(data)

        node_splits = self._create_node_level_splits(data=data, seed=seed)
        if node_splits is None:
            raise ValueError(
                f"Dataset {self.name} uses load_base_dataset but produced no node split; "
                "only node-classification datasets should use that loader."
            )
        # Attach masks before preprocessing so train-mask-aware steps
        # (drop_constant_train_features, final_inductive_zscore) see the split;
        # _create_splits re-attaches the same tensors afterwards.
        data.train_mask, data.val_mask, data.test_mask = node_splits
        data = self._drop_raw_feature_indices(data)
        data = self._apply_feature_preprocessing(data)

        data = self._create_splits(data=data, node_splits=node_splits)

        bundle = wrap_transductive_data(
            data,
            name=self.name,
            edge_set_mode=self.edge_set_mode,
            has_node_features=self.has_node_features,
            has_node_labels=self.has_node_labels,
            has_predefined_node_split=self.has_predefined_node_split,
            has_predefined_edge_split=self.has_predefined_edge_split,
        )
        self._stamp_eval_pref(bundle)
        self.bundle = bundle
        self.data = data
        return bundle

    def _stamp_eval_pref(self, bundle: GraphBundle) -> None:
        """Record this dataset's preferred LP eval protocol on ``bundle.metadata``.

        Consumed by ``MultiGraphLoader`` to pick the eval protocol per dataset
        when the corresponding CLI flag is unset (None).
        """
        if self.preferred_link_pred_eval is not None:
            bundle.metadata["preferred_link_pred_eval"] = self.preferred_link_pred_eval
        if self.preferred_evaluate_head_predictions is not None:
            bundle.metadata["preferred_evaluate_head_predictions"] = bool(
                self.preferred_evaluate_head_predictions
            )
        if self.preferred_eval_split is not None:
            bundle.metadata["preferred_eval_split"] = self.preferred_eval_split
        if self.nc_split_index is not None:
            bundle.metadata["nc_split_index"] = int(self.nc_split_index)

    def _postprocess_bundle(self, bundle: GraphBundle, seed: int) -> GraphBundle:
        """Apply node splits and feature preprocessing to bundle loaders."""
        if bundle.train is None:
            return bundle
        if bundle.metadata.get("features_preprocessed"):
            # Loader-owned feature pipeline: skip the shared PCA / z-score so
            # we do not re-mix columns that the loader already transformed.
            return bundle
        if bundle.has_node_labels:
            node_splits = self._create_node_level_splits(data=bundle.train, seed=seed)
            if node_splits is not None:
                train_mask, val_mask, test_mask = node_splits
                bundle.train.train_mask = train_mask
                bundle.train.val_mask = val_mask
                bundle.train.test_mask = test_mask
        if (
            bundle.has_node_features
            or self.dummy_features
            or self.ignore_features
            or self.ignore_edge_types
        ):
            # Featured inductive KGs attach ``x`` on val/test as well as train.
            # Preprocess every present split that has features (or dummy/ignore)
            # so test does not keep raw PCA-32 while train is z-scored.
            # ``ignore_edge_types`` lives in the same preprocess pass, so
            # featureless KGs must enter this block too or relations stay intact.
            for split_name in ("train", "val", "test"):
                split = getattr(bundle, split_name)
                if split is None:
                    continue
                if (
                    getattr(split, "x", None) is not None
                    or self.dummy_features
                    or self.ignore_features
                    or self.ignore_edge_types
                ):
                    setattr(
                        bundle,
                        split_name,
                        self._apply_feature_preprocessing(split),
                    )
        if bundle.train is not None and getattr(bundle.train, "edge_set", None) is None:
            if is_link_prediction_data(bundle.train):
                bundle.train.edge_set = build_edge_set(bundle.train, bundle.edge_set_mode)
        return bundle
    
    def _create_node_level_splits(
        self,
        data: Data,
        seed: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Create node-level (train, val, test) masks before transforms.

        Returns ``None`` when ``has_node_labels`` is false or ``data.y`` is missing.

        Resolution order:
        1. ``has_predefined_node_split`` and not ``always_generate_split``:
           reuse masks on ``data``, selecting ``nc_split_index`` when they are 2-D.
        2. Otherwise: generate via ``_graphany_mask_splits`` (GraphAny 20-per-class).
           ``nc_split_index`` is the sklearn seed when set; else ``seed``.
        """
        if not self.has_node_labels:
            return None

        if self.has_predefined_node_split and not self.always_generate_split:
            split_index = predefined_split_index(self.nc_split_index)
            n_nodes = int(data.num_nodes)
            n_available = n_predefined_splits_from_masks(
                data.train_mask, data.val_mask, data.test_mask, n_nodes
            )
            if split_index >= n_available:
                raise IndexError(
                    f"{self.name}: nc_split_index={split_index} out of range "
                    f"for {n_available} predefined split(s)"
                )
            train_mask = select_node_split_mask(data.train_mask, split_index, n_nodes)
            val_mask = select_node_split_mask(data.val_mask, split_index, n_nodes)
            test_mask = select_node_split_mask(data.test_mask, split_index, n_nodes)
            return train_mask.clone(), val_mask.clone(), test_mask.clone()

        if getattr(data, "y", None) is None:
            return None

        labels = data.y.numpy()
        num_classes = _num_classes_from_int_labels(data.y)
        num_train_nodes = 20 * num_classes
        return self._graphany_mask_splits(
            n_nodes=data.num_nodes,
            labels=labels,
            num_train_nodes=num_train_nodes,
            seed=graphany_split_seed(seed, self.nc_split_index),
        )

    def _graphland_load_kwargs(self) -> dict:
        params = inspect.signature(self._load_base_dataset_fn).parameters
        kwargs = {}
        if "graphland_different_transform" in params:
            kwargs["graphland_different_transform"] = self.graphland_different_transform
        if "graphland_categorical_as_ordinals" in params:
            kwargs["graphland_categorical_as_ordinals"] = (
                self.graphland_categorical_as_ordinals
            )
        return kwargs

    def _pca_target_dim_for_data(self, data: Data) -> int:
        return effective_pca_target_dim(
            self.pca_target_dim,
            data.num_nodes,
            self.adaptive_pca_node_threshold,
        )

    def _drop_raw_feature_indices(self, data: Data) -> Data:
        """Drop selected raw feature columns before PCA / normalization."""
        if not self.drop_feature_indices:
            return data
        if getattr(data, "x", None) is None:
            raise ValueError(
                f"--drop_feature_indices set but {self.name} has no node features"
            )
        n_feat = int(data.x.shape[1])
        drop = sorted({int(i) for i in self.drop_feature_indices})
        bad = [i for i in drop if i < 0 or i >= n_feat]
        if bad:
            raise ValueError(
                f"drop_feature_indices out of range for {self.name} "
                f"(n_feat={n_feat}): {bad}"
            )
        if len(drop) >= n_feat:
            raise ValueError(
                f"drop_feature_indices would remove all {n_feat} features on {self.name}"
            )
        keep = [i for i in range(n_feat) if i not in set(drop)]
        data.x = data.x[:, keep].contiguous()
        return data

    def _apply_feature_preprocessing(self, data: Data) -> Data:
        """Run :func:`apply_feature_preprocessing` with this dataset's settings."""
        return apply_feature_preprocessing(
            data,
            pca_target_dim=self._pca_target_dim_for_data(data),
            has_node_features=self.has_node_features,
            dummy_features=self.dummy_features,
            ignore_features=self.ignore_features,
            ignore_edge_types=self.ignore_edge_types,
            row_wise_norming=self.row_wise_norming,
            row_norm_mode=self.row_norm_mode,
            pca_before_normalization=self.pca_before_normalization,
            graphland_different_transform=self.graphland_different_transform,
            graphland_categorical_as_ordinals=self.graphland_categorical_as_ordinals,
            drop_constant_train_features=self.drop_constant_train_features,
            final_inductive_zscore=self.final_inductive_zscore,
        )

    def _create_splits(
        self,
        data: Data,
        node_splits: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Data:
        """One-hot encode ``data.y`` and attach node-level train/val/test masks."""
        train_mask, val_mask, test_mask = node_splits
        num_classes = _num_classes_from_int_labels(data.y)
        y_oh = _nan_node_labels_to_dummy(data.y)
        data.y = F.one_hot(y_oh.long(), num_classes).float()
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        return data

    def _graphany_mask_splits(self, n_nodes, labels, num_train_nodes, seed=42):
        """GraphAny ``get_data_split_masks`` (20 labeled train nodes per class).

        https://github.com/DeepGraphLearning/GraphAny/blob/main/graphany/data.py
        """
        assert labels.ndim == 1, "y n_dim must equal 1"
        label_idx = np.arange(n_nodes)
        test_rate_in_labeled_nodes = (len(labels) - num_train_nodes) / len(labels)
        train_idx, test_and_valid_idx = train_test_split(
            label_idx,
            test_size=test_rate_in_labeled_nodes,
            random_state=seed,
            shuffle=True,
            stratify=labels,
        )
        valid_idx, test_idx = train_test_split(
            test_and_valid_idx,
            test_size=0.5,
            random_state=seed,
            shuffle=True,
            stratify=labels[test_and_valid_idx],
        )
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[valid_idx] = True
        test_mask[test_idx] = True

        return train_mask, val_mask, test_mask
