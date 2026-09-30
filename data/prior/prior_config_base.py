"""Shared factories for ``SCMPrior`` hyperparameter config variants.

``build_get_default_fixed_hp`` / ``build_get_default_sampled_hp`` wrap a
complexity-dependent dict and apply CLI/checkpoint overrides.
"""

from __future__ import annotations

from typing import Any, Callable, Optional


def clamp_complexity(complexity: float) -> float:
    if complexity < 0:
        return 0.0
    if complexity > 1.0:
        return 1.0
    return complexity


FIXED_HP_OVERRIDE_KEYS: tuple[str, ...] = (
    "num_nodes",
    "num_features",
    "num_classes",
    "train_size_frac",
    "scm_min_features",
    "prior_type",
    "features_first",
    "data_graph_sampler_type",
    "data_graph_homophily_ratio",
    "data_graph_sbm_noise",
    "data_graph_edge_method",
    "data_graph_kernel_edge_perturb_prob",
    "data_graph_ws_beta_min",
    "data_graph_ws_beta_max",
    "data_graph_ws_label_order_flip_fraction",
    "tree_model",
    "tree_depth_lambda",
    "tree_n_estimators_lambda",
    "balanced",
    "multiclass_ordered_prob",
    "scale_by_max_features",
    "permute_features",
    "permute_labels",
    "x_onehot_y_prob",
    "random_graph_prob",
    "graph_sampler_type",
    "multiclass_type",
    "cat_proportion",
    "cat_ordered_prob",
    "p_cat",
    "cat_ordinal_permute_prob",
    "mlp_dropout_prob",
    "block_wise_dropout",
    "is_causal",
    "num_causes",
    "y_is_effect",
    "in_clique",
    "sort_features",
    "num_layers",
    "hidden_dim",
    "init_std",
    "noise_std",
    "sampling",
    "pre_sample_cause_stats",
    "pre_sample_noise_std",
    "conv_type",
    "graph_conv_ratio",
    "avg_degree",
    "n_lappe_features",
    "degree_features",
    "pagerank_features",
    "bipartite_lp_prob",
    "lp_drop_features_prob",
    "lp_train_size_frac",
    "user_item_ratio",
    "latent_dim_k",
    "popularity_zipf_gamma",
    "feature_informativeness",
    "min_n_first_level_subgraphs",
    "max_n_first_level_subgraphs",
    "min_first_level_degree_ratio",
    "max_first_level_degree_ratio",
    "min_first_level_n_nodes",
    "min_pa_nodes_ratio",
    "max_pa_nodes_ratio",
    "max_pa_degree",
)

SAMPLED_HP_OVERRIDE_KEYS: tuple[str, ...] = (
    "mlp_activations",
    "cat_num_categories",
)


def build_get_default_fixed_hp(
    dict_fn: Callable[[float], dict],
) -> Callable[..., dict]:
    """Build ``get_default_fixed_hp(complexity, **overrides)`` from a dict factory."""

    def get_default_fixed_hp(complexity: float = 1.0, **overrides: Optional[Any]) -> dict:
        cfg = dict_fn(clamp_complexity(complexity))
        for key in FIXED_HP_OVERRIDE_KEYS:
            val = overrides.get(key)
            if val is not None:
                cfg[key] = val
        return cfg

    # Preserve keyword-only override API for introspection / checkpoint tooling.
    get_default_fixed_hp.__doc__ = (
        "Return fixed hyperparameters for SCMPrior. "
        "Pass any key from FIXED_HP_OVERRIDE_KEYS as a keyword to override."
    )
    return get_default_fixed_hp


def build_get_default_sampled_hp(
    dict_fn: Callable[[float], dict],
) -> Callable[..., dict]:
    """Build ``get_default_sampled_hp(complexity, **overrides)`` from a dict factory."""

    def get_default_sampled_hp(complexity: float = 1.0, **overrides: Optional[Any]) -> dict:
        cfg = dict_fn(clamp_complexity(complexity))
        for key in SAMPLED_HP_OVERRIDE_KEYS:
            val = overrides.get(key)
            if val is not None:
                cfg[key] = val
        return cfg

    return get_default_sampled_hp
