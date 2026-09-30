"""Default ``SCMPrior`` hyperparameter distributions (Wander's own graph prior).

Fixed HPs are sampled once per graph (size, SCM type, graph model, …);
sampled HPs cover activations, categorical mixing, and SCM layer noise.
"""

from .activations import ACTIVATION_DIFFICULTY, get_activations
from .prior_config_base import build_get_default_fixed_hp, build_get_default_sampled_hp


def _default_fixed_hp_dict(complexity: float) -> dict:
    """Return fixed HP defaults (distribution specs or literals) for a given complexity in [0, 1]."""
    return {

    # ── SCMPrior ──────────────────────────────────────────────────────────
    # Graph size (number of nodes / sequence length). Mixed log-uniform like avg_degree.
    "num_nodes": {
        "distribution": "mixed_log_uniform",
        "min_first": 200.0,
        "max_first": 3000.0,
        "min_second": 3000.0,
        "max_second": 16000, #8000.0 + complexity * 8000.0,
        "p_first": 0.5,
    },
    # "num_nodes": {
    #     "distribution": "mixed_log_uniform",
    #     "min_first": 100.0,
    #     "max_first": 3000.0,
    #     "min_second": 5000.0,
    #     "max_second": 8000, #8000.0 + complexity * 8000.0,
    #     "p_first": 0.0,
    # },
    # Raw feature count before categorical expansion (rounded to int after sampling).
    "num_features": {
        "distribution": "mixed_log_uniform",
        "min_first": 4.0,
        "max_first": 128.0,
        "min_second": 128.0,
        "max_second": 512.0,
        "p_first": 0.8,
    },
    # "num_features": {
    #     "distribution": "mixed_log_uniform",
    #     "min_first": 4.0,
    #     "max_first": 300.0,
    #     "min_second": 1000.0,
    #     "max_second": 1000.0 + complexity * 500.0,
    #     "p_first": 0.0,
    # },
    
    # Number of classes for classification (rounded to int, at least 2).
    "num_classes": {
        "distribution": "mixed_log_uniform",
        "min_first": 2.0,
        "max_first": 7,
        "min_second": 10.0,
        "max_second": 15.0, # + complexity * 10.0,
        "p_first": 0.5,
    },
    # "num_classes": {
    #     "distribution": "weighted_choice",
    #     "choice_values": [7, 7],
    #     "choice_weights": [0.95, 0.05],
    # },
    # Train split size = round(seq_len * train_size_frac); sampled each graph.
    "train_size_frac": {
        "distribution": "uniform",
        "min": max(0.01,0.1 - complexity * 1),
        "max": 0.5,
    },
    # Minimum SCM feature count for categorical-plan bookkeeping (literal).
    "scm_min_features": 4,
    # Which SCM to use for generating labels (GNNSCM vs TreeSCM)
    "prior_type": {
        "distribution": "weighted_choice",
        "choice_values": ["gnn_scm", "tree_scm"],
        "choice_weights": [1.0, 0.0],
    },
    # ── Generation order ──────────────────────────────────────────────────
    # If True, generate features/labels first, then build graph conditioned
    # on data.  If False, sample the graph first.
    "features_first": {
        "distribution": "weighted_choice",
        "choice_values": [True, False],
        "choice_weights": [0.4, 0.6],
    },
    # ── Data-conditioned graph samplers (used when features_first=True) ──
    # Which sampler to use for building the graph from (X, y)
    "data_graph_sampler_type": {
        "distribution": "weighted_choice",
        "choice_values": [
            "label-sbm",
            "kernel-rbf",
            "kernel-cosine",
            "kernel-inner-product",
            "label-watts-strogatz",
        ],
        "choice_weights": [0.35, 0.2, 0, 0, 0.45],
    },
    # Controls homophily (1.0) vs heterophily (0.0) of the generated graph
    "data_graph_homophily_ratio": {
        "distribution": "bimodal_truncnorm",
        "mu1": 0.8,
        "sigma1": 0.1,
        "mu2": 0.1,
        "sigma2": 0.1,
        "p": 0.7,
        "lower_bound": 0.0,
        "upper_bound": 1.0,
    },
    # Fraction of nodes whose community assignment is randomly flipped (SBM only)
    "data_graph_sbm_noise": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 0.05 + complexity * 0.2,
    },
    # Edge construction method for kernel-based samplers
    "data_graph_edge_method": {
        "distribution": "weighted_choice",
        "choice_values": ["threshold", "bernoulli"],
        "choice_weights": [1,0],
    },
    # Kernel samplers: per-edge prob. to remove an edge and add a random uniform
    # edge (helps LCC size when thresholding fragments the graph). 0 = off.
    "data_graph_kernel_edge_perturb_prob": {
        "distribution": "uniform",
        "min": 0.01,
        "max": 0.05,
    },
    # label-watts-strogatz: WS rewiring probability beta ~ U[min, max]
    "data_graph_ws_beta_min": 0.01,
    "data_graph_ws_beta_max": 0.3,
    # label-watts-strogatz: fraction of nodes (round(frac*n)) with ordering label
    # reassigned to a random other class before sorting the ring. 0 = off.
    "data_graph_ws_label_order_flip_fraction": {
        "distribution": "uniform",
         "min": 0.05,
         "max": 0.25 + complexity * 0.2,
    },
    # ── TreeSCM ───────────────────────────────────────────────────────────
    # Which tree model to use ("xgboost" or "random_forest")
    "tree_model": "xgboost",
    # Poisson lambda for sampling tree depth (higher → deeper trees on average)
    "tree_depth_lambda": 0.5,
    # Poisson lambda for sampling number of estimators (higher → more trees)
    "tree_n_estimators_lambda": 0.5,
    # ── Reg2Cls ───────────────────────────────────────────────────────────
    # Whether to balance class frequencies in the generated classification task
    "balanced": False,
    # Probability that a multiclass task uses ordered (ordinal) class boundaries
    "multiclass_ordered_prob": 0.0,
    # Whether to scale regression targets by max number of features
    "scale_by_max_features": False,
    # Whether to randomly permute feature columns
    "permute_features": True,
    # Whether to randomly permute label assignments
    "permute_labels": True,
    # Probability that X is set to one-hot(permuted y) for trivially easy datasets
    "x_onehot_y_prob": max(0.00, 0.0 - complexity * 4),
    # Per-dataset probability to replace the generated graph with Erdős–Rényi (same n_nodes)
    "random_graph_prob": max(0.01, 0.0 - complexity),
    # Graph topology generator to use for the synthetic graph, when features_first=False
    "graph_sampler_type": {
        "distribution": "weighted_choice",
        "choice_values": [
            "multi-level-sbm-with-pa",
            "geometric",
            "watts-strogatz",
        ],
        "choice_weights": [0.45, 0.3, 0.25],
    },
    # ── Reg2Cls (sampled once per graph) ──────────────────────────────────
    # How to convert regression targets to multiclass labels
    # ("value" = equal-width binning, "rank" = rank-based quantiles)
    "multiclass_type": {
        "distribution": "weighted_choice",
        "choice_values": ["value", "rank"],
        "choice_weights": [0.5, 0.5],
    },
    # Proportion of X columns to convert to categorical features (≈ U[0, 0.5])
    "cat_proportion": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 0.5,
    },
    # Per-dataset probability that a categorical column stays ordinal
    # rather than being permuted + one-hot encoded (≈ U[0, 1])
    "cat_ordered_prob": {
        "distribution": "uniform",
        "min": 0.5,
        "max": 1.0,
    },
    # Probability that any categorical columns are created (1.0 = always; GraphPFN uses 0.2).
    "p_cat": 1.0,
    # After ordinal assignment, randomly permute integer codes (per ordinal column).
    "cat_ordinal_permute_prob": 0.7,
    # ── GNNSCM (sampled once per graph) ───────────────────────────────────
    # Dropout probability for MLP layers in the GNN SCM (≈ U[0, 0.5])
    "mlp_dropout_prob": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 0.5,
    },
    # Whether to apply dropout block-wise (entire layer) rather than element-wise
    "block_wise_dropout": {
        "distribution": "weighted_choice",
        "choice_values": [True, False],
        "choice_weights": [0.5, 0.5],
    },
    # ── GNNSCM and TreeSCM (sampled once per graph) ──────────────────────
    # Whether the causal graph structure is enforced
    # (y depends on causes, not vice versa)
    "is_causal": {
        "distribution": "weighted_choice",
        "choice_values": [True, False],
        "choice_weights": [0.95, 0.05],
    },
    # Number of root-cause features in the SCM
    "num_causes": {
        "distribution": "trunc_norm_log_scaled",
        "max_mean": 12,
        "min_mean": 1,
        "round": True,
        "lower_bound": 1,
        "min_std": 0.01,
        "max_std": 1.0,
    },
    # Whether y is an effect (child) rather than a cause (parent) in the SCM
    "y_is_effect": {
        "distribution": "weighted_choice",
        "choice_values": [True, False],
        "choice_weights": [0.2 + complexity * 0.3, 0.8 - complexity * 0.3],
    },
    # Whether cause features form a clique in the graph
    "in_clique": {
        "distribution": "weighted_choice",
        "choice_values": [True, False],
        "choice_weights": [0.5, 0.5],
    },
    # Whether to sort features by causal ordering before feeding them to the model
    "sort_features": {
        "distribution": "weighted_choice",
        "choice_values": [True, False],
        "choice_weights": [0.5, 0.5],
    },
    # Number of MLP layers in the GNN SCM
    "num_layers": {
        "distribution": "trunc_norm_log_scaled",
        "max_mean": 3 + complexity * 3,
        "min_mean": 0.01 + complexity * 2,
        "round": True,
        "lower_bound": 2,
        "min_std": 0.01,
        "max_std": 1.0,
    },
    # Hidden dimension of MLP layers in the GNN SCM
    "hidden_dim": {
        "distribution": "trunc_norm_log_scaled",
        "max_mean": 32 + complexity * 32,
        "min_mean": 5,
        "round": True,
        "lower_bound": 4,
        "min_std": 0.01,
        "max_std": 1.0,
    },
    # Standard deviation of weight initialisation in the MLP
    "init_std": {
        "distribution": "trunc_norm_log_scaled",
        "max_mean": 15.0,
        "min_mean": 0.01,
        "round": False,
        "lower_bound": 0.0,
        "min_std": 0.01,
        "max_std": 1.0,
    },
    # Standard deviation of additive noise applied to SCM outputs
    "noise_std": {
        "distribution": "trunc_norm_log_scaled",
        "max_mean": 0.05 + complexity * 0.3,
        "min_mean": 0.0001,
        "round": False,
        "lower_bound": 0.0,
        "min_std": 0.01,
        "max_std": 1.0,
    },
    # Distribution used for sampling input (cause) features
    "sampling": {
        "distribution": "weighted_choice",
        "choice_values": ["normal", "mixed", "uniform"],
        "choice_weights": [1.0 / 3, 1.0 / 3, 1.0 / 3],
    },
    # Whether to pre-sample statistics (mean/std) for cause features
    # (fixed per dataset rather than per sample)
    "pre_sample_cause_stats": {
        "distribution": "weighted_choice",
        "choice_values": [True, False],
        "choice_weights": [0.5, 0.5],
    },
    # Whether to pre-sample noise std (fixed per dataset rather than per sample)
    "pre_sample_noise_std": {
        "distribution": "weighted_choice",
        "choice_values": [True, False],
        "choice_weights": [0.5, 0.5],
    },
    # ── Graph structure (sampled once per graph) ──────────────────────────
    # Type of graph convolution layer used in the GNN
    "conv_type": {
        "distribution": "weighted_choice",
        "choice_values": ["gcn", "sage-mean", "sage-min", "sage-max", "gt"],
        "choice_weights": [0.2, 0.2, 0.2, 0.2, 0.2],
    },
    # Fraction of layers that use graph convolution (rest are plain MLP layers)
    "graph_conv_ratio": {
        "distribution": "weighted_choice",
        "choice_values": [0.4, 0.6, 0.8, 1.0],
        "choice_weights": [0.25, 0.25, 0.25, 0.25],
    },
    # Average node degree of the generated graph.
    # Mixture of two log-uniform ranges: one sparse, one dense.
    "avg_degree": {
        "distribution": "mixed_log_uniform",
        "min_first": 2.5,
        "max_first": 20,
        "min_second": 20,
        "max_second": 300,
        "p_first": 0.9,
    },
    # ── Positional / structural features (sampled once per graph) ─────────
    # Number of Laplacian positional-encoding features to add to node inputs
    "n_lappe_features": {
        "distribution": "uniform_int_with_default",
        "min": 2,
        "max": 32,
        "p_default": 1.0,
        "default": 0,
    },
    # Whether to include node degree as an additional input feature
    "degree_features": {
        "distribution": "weighted_choice",
        "choice_values": [True, False],
        "choice_weights": [0.5, 0.5],
    },
    # Whether to include PageRank scores as an additional input feature
    "pagerank_features": {
        "distribution": "weighted_choice",
        "choice_values": [True, False],
        "choice_weights": [0.5, 0.5],
    },
    # ── Link prediction (synthetic LP steps; see SyntheticGraphLoader) ─────
    # Probability that an LP graph is generated as a directed bipartite
    # recommendation graph (via BipartiteLatentFactorSampler); otherwise the
    # usual graph+feature pipeline is reused undirected with labels stripped.
    "bipartite_lp_prob": 0.0,
    # Probability that an LP graph is generated without node features (independent
    # of bipartite_lp_prob). Matches featureless real LP / KG datasets.
    "lp_drop_features_prob": 0.5,
    # Fraction of edges kept as visible/context (train); the rest become held-out
    # query (test) edges that the synthetic LP step trains on.
    "lp_train_size_frac": {
        "distribution": "uniform",
        "min": 0.7,
        "max": 0.9,
    },
    # Bipartite only: ratio of users to items (U / I).
    "user_item_ratio": {
        "distribution": "uniform",
        "min": 0.3,
        "max": 3.0,
    },
    # Bipartite only: shared latent-factor dimensionality K (rounded to int).
    "latent_dim_k": {
        "distribution": "uniform",
        "min": 4.0,
        "max": 32.0,
    },
    # Bipartite only: Zipf exponent for item popularity bias (heavier tail when
    # closer to 1; > 1 required by numpy's zipf).
    "popularity_zipf_gamma": {
        "distribution": "uniform",
        "min": 1.5,
        "max": 2.5,
    },
    # Bipartite only: scale of the shared latent->feature projection relative to
    # observation noise (higher = more informative features).
    "feature_informativeness": {
        "distribution": "uniform",
        "min": 0.5,
        "max": 2.0,
    },

}


get_default_fixed_hp = build_get_default_fixed_hp(_default_fixed_hp_dict)


def _default_sampled_hp_dict(complexity: float) -> dict:
    """Return sampled HP defaults (meta-distribution specs) for a given complexity in [0, 1]."""
    activations = get_activations()

    base_min, base_max = -5.0, 5.0
    logit_shift = 10.0
    logit_ranges = []
    for act in activations:
        d = ACTIVATION_DIFFICULTY.get(act, 0.0)
        logit_ranges.append([base_min , base_max - d * logit_shift + complexity * d * logit_shift])

    return {
        # ── Multi-draw HPs (meta-distributions, sampled per layer / per column) ──
        # Activation functions used in MLP layers of the GNN SCM.
        # Logits are drawn once per graph (with per-activation ranges that
        # depend on complexity), then one activation *constructor* is chosen
        # per layer via softmax + multinomial.
        "mlp_activations": {
            "distribution": "meta_choice_mixed",
            "choice_values": activations,
            "choice_weight_ranges": logit_ranges,
        },
        # Number of categories per categorical column.
        # Distribution params are drawn once per graph, then one value is drawn
        # per categorical column.
        "cat_num_categories": {
            "distribution": "meta_trunc_norm_log_scaled",
            "max_mean": 5,
            "min_mean": 2,
            "round": True,
            "lower_bound": 2,
            "min_std": 0.01,
            "max_std": 1.0,
        },
    }


get_default_sampled_hp = build_get_default_sampled_hp(_default_sampled_hp_dict)


DEFAULT_SAMPLED_HP = get_default_sampled_hp()
