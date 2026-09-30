"""Convert synthetic prior output (PyG graph + tensors) to PyG Data objects."""

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data

from .dataset import apply_feature_preprocessing


def prior_to_pyg_data(
    graph: Data,
    X: Tensor,
    y: Tensor,
    d: int,
    train_size: int,
    pca_target_dim: int = 32,
    row_wise_norming: bool = False,
    row_norm_mode: str = "l2",
    dummy_features: bool = False,
    ignore_features: bool = False,
    pca_before_normalization: bool = True,
    graphland_different_transform: bool = False,
    graphland_categorical_as_ordinals: bool = False,
    drop_constant_train_features: bool = False,
    final_inductive_zscore: bool = False,
) -> Data:
    """Convert a single synthetic prior dataset to a PyG Data object.

    Parameters
    ----------
    graph : Data
        Graph structure produced by GraphSampler (nodes = samples).
    X : Tensor
        Node features of shape ``(num_nodes, d)``.
    y : Tensor
        Integer class labels of shape ``(num_nodes,)``.
    d : int
        Number of feature columns in *X* (equals ``X.shape[1]``).
    train_size : int
        Positional boundary: nodes ``[0, train_size)`` are train,
        ``[train_size, num_nodes)`` are test (always-unknown labels).
    pca_target_dim : int
        Target dimensionality for PCA feature reduction.

    Returns
    -------
    Data
        PyG Data object ready for ``_train_step`` / ``BatchLoaderNodeClassification``.
    """
    num_nodes = graph.num_nodes
    edge_index = graph.edge_index.long()

    x = X.float()

    y_long = y.long()
    is_regression = (y.float() != y_long.float()).any() or (y_long.min() < 0)
    if is_regression:
        n_cls = max(2, min(10, int(y.unique().numel() ** 0.5)))
        quantiles = torch.linspace(0, 1, n_cls + 1, device=y.device)[1:-1]
        thresholds = torch.quantile(y.float(), quantiles)
        y_long = torch.bucketize(y.float(), thresholds)
    num_classes = int(y_long.max().item()) + 1
    y_onehot = F.one_hot(y_long, num_classes).float()

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:train_size] = True
    test_mask = ~train_mask

    data = Data(
        x=x,
        y=y_onehot,
        edge_index=edge_index,
        num_nodes=num_nodes,
        train_mask=train_mask,
        test_mask=test_mask,
    )
    data.supervise_non_train_nodes = True

    return apply_feature_preprocessing(
        data,
        pca_target_dim=pca_target_dim,
        has_node_features=True,
        dummy_features=dummy_features,
        ignore_features=ignore_features,
        row_wise_norming=row_wise_norming,
        row_norm_mode=row_norm_mode,
        pca_before_normalization=pca_before_normalization,
        graphland_different_transform=graphland_different_transform,
        graphland_categorical_as_ordinals=graphland_categorical_as_ordinals,
        drop_constant_train_features=drop_constant_train_features,
        final_inductive_zscore=final_inductive_zscore,
    )


def prior_to_pyg_lp_data(
    X: Optional[Tensor],
    lp_meta: Dict[str, Any],
    pca_target_dim: int = 32,
    row_wise_norming: bool = False,
    row_norm_mode: str = "l2",
    dummy_features: bool = False,
    ignore_features: bool = False,
    pca_before_normalization: bool = True,
    graphland_different_transform: bool = False,
    graphland_categorical_as_ordinals: bool = False,
    drop_constant_train_features: bool = False,
    final_inductive_zscore: bool = False,
) -> Data:
    """Convert a synthetic link-prediction sample to a PyG Data object.

    The edges in ``lp_meta`` are split into context (train) and held-out query
    (test) edges. The returned ``Data`` exposes only the context graph as
    ``edge_index`` (so the model never sees the answer), the feature pipeline is
    applied as in :func:`prior_to_pyg_data`, ``edge_set`` contains all positives
    (both directions for undirected) for negative-sampling filtering, and
    ``lp_test_edges`` (``[2, E_test]`` head/tail pairs) are the queries the
    synthetic LP step trains on.

    Parameters
    ----------
    X : Tensor or None
        Node features of shape ``(num_nodes, d)``, or ``None`` when
        ``lp_meta["drop_features"]`` is True.
    lp_meta : dict
        Output of :meth:`SCMPrior._get_batch_lp` with keys ``edge_index``,
        ``is_bipartite``, ``candidate_offset``, ``num_nodes``,
        ``lp_train_size_frac``, and ``drop_features``.
    """
    drop_features = bool(lp_meta.get("drop_features", False))
    has_raw_features = X is not None and not drop_features

    num_nodes = int(lp_meta["num_nodes"])
    ei = lp_meta["edge_index"].long().cpu()
    is_bipartite = bool(lp_meta["is_bipartite"])
    candidate_offset = int(lp_meta["candidate_offset"])
    frac = float(lp_meta["lp_train_size_frac"])

    # Canonical (deduplicated) edges. Bipartite edges are already directed
    # user -> item; undirected edges keep one orientation (u < v).
    if is_bipartite:
        canon = ei
    else:
        u, v = ei[0], ei[1]
        keep = u < v
        canon = torch.stack([u[keep], v[keep]], dim=0)

    E = canon.shape[1]
    if E == 0:
        canon = torch.tensor([[0], [min(1, num_nodes - 1)]], dtype=torch.long)
        E = 1

    perm = torch.randperm(E)
    if E > 1:
        n_train = int(round(E * frac))
        n_train = max(1, min(E - 1, n_train))
    else:
        n_train = 1
    train_e = canon[:, perm[:n_train]]
    test_e = canon[:, perm[n_train:]]
    if test_e.shape[1] == 0:
        test_e = train_e[:, :1]

    if is_bipartite:
        ctx_ei = train_e
    else:
        ctx_ei = torch.cat([train_e, train_e.flip(0)], dim=1)

    if has_raw_features:
        data = Data(x=X.float(), edge_index=ctx_ei, num_nodes=num_nodes)
    else:
        data = Data(edge_index=ctx_ei, num_nodes=num_nodes)

    data = apply_feature_preprocessing(
        data,
        pca_target_dim=pca_target_dim,
        has_node_features=has_raw_features,
        dummy_features=dummy_features,
        ignore_features=ignore_features or not has_raw_features,
        row_wise_norming=row_wise_norming,
        row_norm_mode=row_norm_mode,
        ensure_edge_type=True,
        pca_before_normalization=pca_before_normalization,
        graphland_different_transform=graphland_different_transform,
        graphland_categorical_as_ordinals=graphland_categorical_as_ordinals,
        drop_constant_train_features=drop_constant_train_features,
        final_inductive_zscore=final_inductive_zscore,
    )
    data.is_link_prediction = True

    all_canon = torch.cat([train_e, test_e], dim=1)
    if is_bipartite:
        pairs = all_canon
    else:
        pairs = torch.cat([all_canon, all_canon.flip(0)], dim=1)
    data.edge_set = {
        (int(h), 0, int(t))
        for h, t in zip(pairs[0].tolist(), pairs[1].tolist())
    }

    if is_bipartite:
        data.anygraph_bipartite = True
        data.anygraph_candidate_offset = candidate_offset
    else:
        data.undirected_lp = True

    data.lp_test_edges = test_e

    return data
