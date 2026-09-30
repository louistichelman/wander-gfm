"""Argument definitions for Wander."""

import argparse
from typing import Optional

from data.prior.prior_config_loader import PRIOR_CONFIG_NAMES


def get_argument_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser for Wander.
    
    Returns:
        Configured ArgumentParser with all supported arguments.
    """
    parser = argparse.ArgumentParser(
        description="Wander training and evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ==================== Model Architecture ====================
    model = parser.add_argument_group("model architecture")
    model.add_argument(
        "--wander_walk_len",
        type=int,
        default=32,
        help="Length of each random walk. Must be <= --wander_max_walk_len",
    )
    model.add_argument(
        "--wander_max_walk_len",
        type=int,
        default=128,
        help=(
            "Max walk length; sizes emb_anon_node/type tables. Runtime length is "
            "--wander_walk_len (or adaptive). Must be >= --wander_walk_len. "
        ),
    )
    model.add_argument(
        "--wander_adaptive_walks",
        action="store_true",
        default=False,
        help="Scale runtime walk_num/walk_len per graph from --wander_walk_num and --wander_max_walk_len by num_nodes tiers (>=8000 full, >=1000 //2, else //4).",
    )
    model.add_argument(
        "--no_wander_adaptive_walks",
        dest="wander_adaptive_walks",
        action="store_false",
        help="Disable adaptive walk scaling; use --wander_walk_num / --wander_walk_len as-is.",
    )
    model.add_argument(
        "--wander_record_neighbors",
        action="store_true",
        default=False,
        help="Record neighbors during walks",
    )
    model.add_argument("--wander_refinements", type=int, default=6, help="Number of refinement iterations")
    model.add_argument("--wander_attention_scatter", action="store_true", default=True, help="Use attention-weighted scatter")
    model.add_argument("--no_wander_attention_scatter", dest="wander_attention_scatter", action="store_false", help="Disable attention-weighted scatter")
    model.add_argument("--wander_attention_scatter_n_heads", type=int, default=4, help="Number of attention heads for scatter")
    model.add_argument("--wander_additive_refinement", action="store_true", default=True, help="Use additive refinement updates")
    model.add_argument("--no_wander_additive_refinement", dest="wander_additive_refinement", action="store_false", help="Disable additive refinement updates")
    model.add_argument(
        "--wander_init_norm",
        action="store_true",
        default=True,
        help="RMSNorm feature/label (and thinking-feature) embeddings at init before the refinement loop",
    )
    model.add_argument(
        "--no_wander_init_norm",
        dest="wander_init_norm",
        action="store_false",
        help="Disable init RMSNorm on feature/label embeddings",
    )
    model.add_argument(
        "--wander_pre_norm",
        action="store_true",
        default=False,
        help=(
            "Outer Pre-LN on additive RW refinement: read from RMSNorm(h), write h <- h + Delta. "
            "Only used when --wander_additive_refinement is on."
        ),
    )
    model.add_argument(
        "--no_wander_pre_norm",
        dest="wander_pre_norm",
        action="store_false",
        help="Disable outer Pre-LN on additive RW refinement updates",
    )
    model.add_argument("--wander_embed_first_only", action="store_true", default=False, help="Only embed walks in the first refinement")
    model.add_argument("--wander_embedding_tying", action="store_true", default=False, help="Tie embeddings across refinements")
    model.add_argument("--wander_param_tying_across_refinements", action="store_true", default=False, help="Tie parameters across refinements")
    model.add_argument("--wander_param_tying_across_channels", action="store_true", default=False, help="Tie parameters within refinements (node/feature/label)")
    model.add_argument(
        "--wander_untie_attention_across_tasks",
        action="store_true",
        default=False,
        help=(
            "Use separate global (context) attention parameters for node classification "
            "vs link prediction. Default shares one net_global across both tasks."
        ),
    )
    model.add_argument(
        "--no_wander_untie_attention_across_tasks",
        dest="wander_untie_attention_across_tasks",
        action="store_false",
        help="Share global attention parameters across node classification and link prediction",
    )
    model.add_argument("--wander_fixed_rw_across_refinements", action="store_true", default=False, help="Reuse the same random walks across all refinement steps")
    model.add_argument("--wander_hidden_dim", type=int, default=128, help="Hidden dimension for Wander")
    model.add_argument("--wander_dtype", type=str, default="float32", help="Data type for Wander parameters")
    model.add_argument("--wander_net", type=str, choices=["gru", "attention"], default="gru", help="Sequence network type for processing random walks")
    model.add_argument("--wander_n_layers", type=int, default=1, help="Number of layers in the sequence network (GRU layers or attention layers)")
    model.add_argument("--wander_net_n_heads", type=int, default=4, help="Number of attention heads for the sequence network (only used when --wander_net=attention). Defaults to hidden_dim // 64")
    model.add_argument(
        "--wander_randomize_feat_columns",
        action="store_true",
        default=True,
        help="Randomly permute node feature columns each forward; with --wander_test_samples>1, use a different permutation per ensemble draw (node cls and link pred).",
    )
    model.add_argument(
        "--no_wander_randomize_feat_columns",
        dest="wander_randomize_feat_columns",
        action="store_false",
        help="Disable per-forward random permutation of node feature columns.",
    )
    model.add_argument(
        "--wander_randomize_label_columns",
        action="store_true",
        default=True,
        help="Randomly permute one-hot label class columns each forward; with --wander_test_samples>1, use a different permutation per ensemble draw (node cls and link pred).",
    )
    model.add_argument(
        "--no_wander_randomize_label_columns",
        dest="wander_randomize_label_columns",
        action="store_false",
        help="Disable per-forward random permutation of label class columns.",
    )
    model.add_argument("--wander_intra_node_attention", action="store_true", default=True, help="Enable intra-node self-attention across features/labels/node")
    model.add_argument("--no_wander_intra_node_attention", dest="wander_intra_node_attention", action="store_false", help="Disable intra-node self-attention across features/labels/node")
    model.add_argument("--wander_intra_node_attention_n_heads", type=int, default=4, help="Number of attention heads for intra-node self-attention")
    model.add_argument(
        "--wander_rotary_emb_intra_node_attention",
        action="store_true",
        default=True,
        help="Apply rotary positional embeddings to Q/K in intra-node self-attention (token order: node, features, label). Requires --wander_intra_node_attention.",
    )
    model.add_argument(
        "--no_wander_rotary_emb_intra_node_attention",
        dest="wander_rotary_emb_intra_node_attention",
        action="store_false",
        help="Disable rotary positional embeddings in intra-node self-attention.",
    )
    model.add_argument(
        "--wander_intra_node_ffn",
        action="store_true",
        default=False,
        help=(
            "After intra-node attention, apply a Pre-LN FFN residual (same pattern as "
            "global SequenceAttention). Default off matches historical attn-only IntraNode."
        ),
    )
    model.add_argument("--wander_inter_node_chunksize", type=int, default=128, help="Max attributes (features/classes) to process at once through sequence nets and local attention")
    model.add_argument("--wander_intra_node_chunksize", type=int, default=2048, help="Max nodes (B*N) to process through intra-node attention at once")
    model.add_argument("--wander_checkpoint_refinements", action="store_true", default=True, help="Gradient-checkpoint each refinement iteration to trade compute for ~num_refinementsx memory savings")
    model.add_argument("--no_wander_checkpoint_refinements", dest="wander_checkpoint_refinements", action="store_false", help="Disable gradient checkpointing over refinement iterations")
    model.add_argument("--wander_checkpoint_sublayers", action="store_true", default=True, help="Gradient-checkpoint sublayer calls (sequence nets, intra-node chunks) to reduce activation memory")
    model.add_argument("--no_wander_checkpoint_sublayers", dest="wander_checkpoint_sublayers", action="store_false", help="Disable gradient checkpointing inside refinement sublayers")
    model.add_argument("--wander_layerscale", type=float, default=0.0, help="LayerScale init value for additive refinement updates. 0 = disabled, e.g. 0.1 = scale updates by learnable per-dim vector initialized to 0.1")
    model.add_argument("--wander_global_attention_update", action="store_true", default=True, help="Apply a global self-attention update over all (post-prune) nodes in each refinement step. Can be combined with --wander_rw_update; runs first when both are on.")
    model.add_argument("--no_wander_global_attention_update", dest="wander_global_attention_update", action="store_false", help="Disable the global self-attention update in each refinement step.")
    model.add_argument(
        "--wander_global_kv_neighbors",
        type=int,
        default=None,
        help=(
            "Max query-relation neighbors kept as global-attention K/V per "
            "link-prediction row. The query node (and thinking rows) are always "
            "kept; node-role embeddings are unchanged. Default: no cap."
        ),
    )
    model.add_argument("--wander_rw_update", action="store_true", default=True, help="Apply the random-walk based inter-node update in each refinement step. Can be combined with --wander_global_attention_update; runs after global attention when both are on.")
    model.add_argument("--no_wander_rw_update", dest="wander_rw_update", action="store_false", help="Disable the random-walk based inter-node update.")
    model.add_argument(
        "--wander_disable_rw_graph_prune",
        action="store_true",
        default=False,
        help="Keep the full graph instead of pruning to visited/query nodes.",
    )
    model.add_argument(
        "--wander_feature_embedding_mlp",
        action="store_true",
        default=True,
        help="Use a 2-layer MLP for per-feature embedding. When disabled, a single Linear layer is used.",
    )
    model.add_argument(
        "--no_wander_feature_embedding_mlp",
        dest="wander_feature_embedding_mlp",
        action="store_false",
        help="Use a single Linear layer for per-feature embedding instead of the 2-layer MLP.",
    )
    model.add_argument("--wander_feature_groups", type=int, default=1, help="If >1, embed groups of this many feature values jointly (one Linear/MLP call per group). Pads the last group with zeros if num_features is not divisible.")
    model.add_argument("--wander_concat_walks", action="store_true", default=False, help="Concatenate all walks per refinement step into one long sequence so nodes across walks can attend to each other")
    model.add_argument(
        "--wander_thinking_features",
        type=int,
        default=0,
        help="Number of learnable thinking-feature slots appended after real features on each graph node.",
    )
    model.add_argument(
        "--wander_thinking_rows",
        type=int,
        default=0,
        help="Number of edgeless virtual train nodes whose tokens are initialized from learnable row embeddings.",
    )
    model.add_argument(
        "--wander_only_forward_edges_when_inverses_added",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Only relevant with --add_inverse_edges_KGs. If set, the random-walk "
            "edge-type parser only uses forward (along-walk / downstream) edges, so "
            "each step is labelled with the relation pointing along the walk, and the "
            "direction embedding is dropped. If not set (default), use the original "
            "random forward/reverse candidate selection together with the direction "
            "embedding."
        ),
    )

    # ==================== Synthetic Prior ====================
    prior = parser.add_argument_group("synthetic prior")
    prior.add_argument("--synthetic_prior", action="store_true", default=False, help="Train on synthetically generated graphs (on-the-fly, one per step)")
    prior.add_argument(
        "--prior_config",
        type=str,
        default="default",
        choices=list(PRIOR_CONFIG_NAMES),
        help=(
            "Synthetic-prior config when --synthetic_prior is enabled. "
            "default uses SCMPrior. "
            "graphpfn / nodepfn sample graphs from the official GraphPFN / NodePFN "
            "prior code (classification only; LP reuses those graphs undirected). "
            "graphpfn needs DGL in this env; nodepfn needs gpytorch and ConfigSpace."
        ),
    )
    prior.add_argument(
        "--features_first_only",
        action="store_true",
        default=False,
        help=(
            "When training with --synthetic_prior, force features_first=True for every "
            "synthetic graph (equivalent to setting its sampling probability to 1)."
        ),
    )
    prior.add_argument(
        "--prior_max_node_cap",
        type=int,
        default=None,
        help=(
            "When training with --synthetic_prior, overwrite num_nodes.max_second in the "
            "SCMPrior fixed-HP config (hard cap on sampled graph size). Applied via "
            "runtime overrides, so it also takes effect on --resume_checkpoint even if "
            "the run's saved prior_config.py still lists a larger max."
        ),
    )
    prior.add_argument(
        "--synthetic_link_pred_prob",
        type=float,
        default=0.0,
        help="Probability that a synthetic training step is a link-prediction step "
             "(vs node classification). 0.0 = node-cls only. LP steps generate "
             "either a directed bipartite graph (prob. bipartite_lp_prob in "
             "prior_config) or an undirected label-stripped graph.",
    )
    prior.add_argument("--prior_complexity_start", type=float, default=1.0, help="Complexity of the synthetic prior at the start of training (1.0 = full prior, used throughout paper training)")
    prior.add_argument("--prior_complexity_adapt", action="store_true", default=False, help="Adapt the complexity of the synthetic prior during training based on validation accuracy")
    prior.add_argument("--synthetic_prefetch", type=int, default=10, help="Number of synthetic graphs to pre-generate in the background during training/evaluation")
    prior.add_argument(
        "--synthetic_prefetch_workers",
        type=int,
        default=1,
        help="Background processes (or threads if --no_synthetic_prefetch_mp) that "
             "generate synthetic graphs. Raise this when GraphPFN/NodePFN sampling "
             "is slower than a training step (default 1 keeps the original "
             "single-producer loader).",
    )
    prior.add_argument(
        "--synthetic_prefetch_mp",
        action="store_true",
        default=True,
        help="Use spawn processes for synthetic prefetch workers (needed for "
             "GraphPFN: sampling is Python/DGL-heavy and threads share the GIL).",
    )
    prior.add_argument(
        "--no_synthetic_prefetch_mp",
        dest="synthetic_prefetch_mp",
        action="store_false",
        help="Use threads instead of processes for synthetic prefetch workers.",
    )
    prior.add_argument(
        "--synthetic_prefetch_keep_workers",
        action="store_true",
        default=True,
        help="Keep spawn prefetch workers alive across epochs (avoids DGL "
             "re-import + cold-sample cost). Needs enough RAM for idle workers "
             "during eval (GraphPFN PT1 uses 128G).",
    )
    prior.add_argument(
        "--no_synthetic_prefetch_keep_workers",
        dest="synthetic_prefetch_keep_workers",
        action="store_false",
        help="Shut down spawn prefetch workers at the end of every training epoch.",
    )
    prior.add_argument("--synthetic_val_steps", type=int, default=100, help="Number of synthetic graphs to evaluate per epoch for synthetic validation")
    prior.add_argument(
        "--synthetic_val_seed",
        type=int,
        default=12345,
        help=(
            "RNG seed used while drawing the synthetic validation tasks. The "
            "RNG is seeded to this before each synthetic-val pass and restored "
            "afterwards, so every epoch scores the SAME set of synthetic tasks. "
            "Without it, epoch-to-epoch movement in syn_acc is dominated by "
            "task resampling noise (route / class count / label noise / "
            "imbalance all vary per draw) rather than by model quality."
        ),
    )

    # ==================== Data Preprocessing ====================
    data = parser.add_argument_group("data preprocessing")
    data.add_argument("--data_dir", type=str, default="./data", help="Directory for dataset storage")
    data.add_argument(
        "--pca_target_dim",
        type=int,
        default=64,
        help="PCA target dimension for real-world datasets (node classification; also LP unless --pca_target_dim_link is set).",
    )
    data.add_argument(
        "--pca_target_dim_link",
        type=int,
        default=None,
        help="PCA target dimension for real-world link-prediction datasets (default: same as --pca_target_dim).",
    )
    data.add_argument(
        "--pca_target_dim_node_syn",
        type=int,
        default=64,
        help="PCA target dimension for synthetic node-classification graphs.",
    )
    data.add_argument(
        "--pca_target_dim_link_syn",
        type=int,
        default=32,
        help="PCA target dimension for synthetic link-prediction graphs.",
    )
    data.add_argument(
        "--adaptive_pca_node_threshold",
        type=int,
        default=0,
        help=(
            "When > 0, halve PCA target dims for graphs with more nodes than this "
            "threshold."
        ),
    )
    data.add_argument("--ignore_features", action="store_true", default=False, help="Ignore node features of datasets")
    data.add_argument(
        "--ignore_edge_types",
        action="store_true",
        default=False,
        help=(
            "Collapse all edges to a single relation type (edge_type=0, num_relations=1). "
            "Use to ablate multi-relational structure on knowledge graphs."
        ),
    )
    data.add_argument(
        "--dummy_features",
        action="store_true",
        default=False,
        help="Replace node features with a single constant column of zeros (one feature per node)",
    )
    data.add_argument(
        "--row_wise_norming",
        action="store_true",
        default=False,
        help=(
            "Use per-row normalization after PCA. Default is transductive column "
            "z-score (statistics computed over all nodes). Use --row_norm_mode to "
            "choose L1 vs L2 when this flag is set."
        ),
    )
    data.add_argument(
        "--row_norm_mode",
        type=str,
        choices=["l1", "l2"],
        default="l2",
        help=(
            "Row normalization type when --row_wise_norming is set: l1 or l2 "
            "(unit L2 norm, default)."
        ),
    )
    data.add_argument(
        "--pca_before_normalization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When True: smoothing -> PCA -> normalization. "
            "When False (default): smoothing -> normalization -> PCA (GraphPFN eval order)."
        ),
    )
    data.add_argument(
        "--graphland_categorical_as_ordinals",
        action="store_true",
        default=True,
        help=(
            "When --graphland_different_transform: load GraphLand categorical columns "
            "as integer ordinals instead of one-hot at load time."
        ),
    )
    data.add_argument(
        "--no_graphland_categorical_as_ordinals",
        dest="graphland_categorical_as_ordinals",
        action="store_false",
        help="One-hot GraphLand categorical columns at load time (legacy).",
    )
    data.add_argument(
        "--graphland_different_transform",
        action="store_true",
        default=True,
        help=(
            "True (default): raw GraphLand load + per-type quantile transforms "
            "(no column z-score) then PCA. False: legacy GraphLand load transforms "
            "+ global pipeline z-score."
        ),
    )
    data.add_argument(
        "--no_graphland_different_transform",
        dest="graphland_different_transform",
        action="store_false",
        help="Use legacy GraphLand load transforms + global pipeline z-score.",
    )
    data.add_argument(
        "--drop_constant_train_features",
        action="store_true",
        default=True,
        help="Drop feature columns that are constant on training nodes (final pipeline step).",
    )
    data.add_argument(
        "--no_drop_constant_train_features",
        dest="drop_constant_train_features",
        action="store_false",
        help="Keep feature columns that are constant on training nodes.",
    )
    data.add_argument(
        "--drop_feature_indices",
        type=int,
        nargs="*",
        default=None,
        metavar="I",
        help=(
            "0-based raw feature column indices to drop before PCA/normalization "
            "(feature ablation). Empty/omitted = drop none."
        ),
    )
    data.add_argument(
        "--final_inductive_zscore",
        action="store_true",
        default=True,
        help=(
            "Append a per-column z-score as the very last feature step (after "
            "PCA / per-type transforms / drop-constant), using statistics from "
            "training nodes only and clipping to +-100. Mimics the input "
            "normalization LimiX/GraphPFN applies inside the model, for both "
            "synthetic and real datasets. Falls back to all-node statistics "
            "when no train mask is available."
        ),
    )
    data.add_argument(
        "--no_final_inductive_zscore",
        dest="final_inductive_zscore",
        action="store_false",
        help="Disable the final inductive per-column z-score feature step.",
    )
    data.add_argument(
        "--add_inverse_edges_KGs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For knowledge graphs only: add explicit inverse edges with new relation types (doubles num_relations).",
    )

    # ==================== Training / Task ====================
    train = parser.add_argument_group("training/task")
    train.add_argument(
        "--wander_walk_num",
        type=int,
        default=16,
        help=(
            "Base walk count per refinement for training, and for eval when "
            "--wander_eval_walk_num is unset."
        ),
    )
    train.add_argument("--max_epochs", type=int, default=25, help="Maximum number of training epochs")
    train.add_argument(
        "--patience",
        type=int,
        default=0,
        help=(
            "Early-stop after this many consecutive epochs without a validation "
            "score increase (same metric as --checkpoint_metric). 0 (default) "
            "disables early stopping and trains to --max_epochs. Requires inline "
            "validation (incompatible with --skip_eval)."
        ),
    )
    train.add_argument("--batch_size_node", type=int, default=16, help="Batch size for native node-classification loaders (real and synthetic).")
    train.add_argument(
        "--nc_train_query_frac",
        type=float,
        default=0.0,
        help=(
            "Real-world NC training only: fraction of train nodes whose labels are "
            "hidden each forward (in addition to the batch). Query count is "
            "max(min(int(n_train * frac), --nc_train_query_cap), batch_size). "
            "Walks still start from the loader batch; CE is on hidden train nodes "
            "that survive RW prune. 0 (default) disables this unless the cap is "
            "also set; both frac and cap must be > 0."
        ),
    )
    train.add_argument(
        "--nc_train_query_cap",
        type=int,
        default=0,
        help=(
            "Cap on extra real-world NC train queries per forward (see "
            "--nc_train_query_frac). 0 disables the expansion."
        ),
    )
    train.add_argument(
        "--batch_size_link",
        type=int,
        default=4,
        help="Link-prediction train batch size: queries per forward (may mix relations).",
    )
    train.add_argument(
        "--batch_per_epoch",
        type=int,
        default=2000,
        help=(
            "Number of training batches per epoch. Default sampling is with "
            "replacement. With --train_without_replacement this is only the "
            "step count: a shuffled deck is consumed without replacement and "
            "continues across epochs, reshuffling only when it is exhausted."
        ),
    )
    train.add_argument(
        "--train_without_replacement",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Consume real-graph train queries/nodes from a shuffled deck "
            "without replacement. The deck continues across epochs and is "
            "reshuffled only when empty (last batch of a pass may be shorter). "
            "Incompatible with class-balanced NC batches and unused for "
            "contrastive LP / synthetic loaders."
        ),
    )
    train.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    train.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    train.add_argument(
        "--node_cls_random_training_batches",
        action="store_true",
        default=True,
        help="For node classification, draw training batches uniformly from train nodes instead of class-balanced sampling (real and synthetic train).",
    )
    train.add_argument(
        "--no_node_cls_random_training_batches",
        dest="node_cls_random_training_batches",
        action="store_false",
        help="Use class-balanced sampling for node-classification training batches.",
    )
    train.add_argument("--amp_dtype", type=str, choices=["float32", "bfloat16", "float16"], default="float32", help="AMP autocast dtype (float32 to disable)")
    train.add_argument("--num_negatives", type=int, default=512, help="Number of negative samples per query")
    train.add_argument("--adversarial_temperature", type=float, default=1.0, help="Temperature for adversarial negative reweighting (0 = uniform)")
    train.add_argument("--info_nce_temperature", type=float, default=0.0, help="Temperature for additive InfoNCE loss term (0 = disabled)")
    train.add_argument(
        "--num_graphs_batch",
        type=int,
        default=1,
        help="Number of graphs per optimizer step (global). Each graph contributes "
             "min(batch_size_*, #train_nodes/edges) queries; per-graph losses are "
             "averaged before a single backward pass. With multi-GPU (torchrun), must "
             "divide evenly by the number of processes so each rank runs a slice. "
             "Works with synthetic, real, or mixed training.",
    )
    train.add_argument(
        "--real_world_graph_prob",
        type=float,
        default=0.0,
        help="When training with both --synthetic_prior and --train_datasets, "
             "probability that a step draws from the real datasets instead of the "
             "synthetic prior. 0.0 = synthetic only, 1.0 = real only. Ignored "
             "outside of mixed mode.",
    )
    train.add_argument(
        "--train_datasets",
        type=str,
        nargs="+",
        default=[],
        help="Training dataset names (e.g. CORA, FB15K_237). Registry names only; optional NAME:version.",
    )
    train.add_argument(
        "--nc_split_index",
        type=int,
        default=None,
        help=(
            "Node-classification split to train and evaluate. Same protocol as "
            "zero-shot NC eval (data/nc_splits.py): official mask column, or "
            "GraphAny sklearn seed 0..4 when there is no official split. Default: "
            "first official split, or --seeds for GraphAny-generated splits. "
            "Finetune / eval queues pass one index per job (NC_SPLITS=all)."
        ),
    )
    train.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Experiment seeds (default: config.SEEDS, currently [0]). NC / ordinary "
            "LP queues pass 0 1 2 and report mean ± std. For UniLP sampled-recall "
            "eval each seed is the cached split run index (split{seed}_10_20.pt)."
        ),
    )

    # ==================== Checkpoint / Logging ====================
    ckpt = parser.add_argument_group("checkpoint/logging")
    ckpt.add_argument("--save_load_path", type=str, default="./checkpoints", help="Path for saving/loading model checkpoints")
    ckpt.add_argument("--save_checkpoint_interval", type=int, default=1, help="Save model weights every N epochs. 0 means no periodic saving.")
    ckpt.add_argument("--run_name", type=str, default=None, help="Fixed experiment/checkpoint directory name. Overrides auto-generated base_name when set.")
    ckpt.add_argument("--resume_checkpoint", type=str, default=None, help="Path to a full checkpoint to resume training from (restores model, optimizer, and epoch).")
    ckpt.add_argument("--init_checkpoint", type=str, default=None, help="Path to a .pt checkpoint file to initialize model weights from before training (fine-tuning).")
    ckpt.add_argument(
        "--reset_rw_parameters",
        action="store_true",
        default=False,
        help=(
            "With --init_checkpoint only: instantiate Wander from current CLI, keep default-initialized "
            "random-walk parameters (net_rw, from_*/to_*, walk position embeddings, scatter logits, "
            "walk_emb_gate_*, pre_norm_*, and ls_* LayerScale used for RW scatter merges), and load every other "
            "parameter from the checkpoint. The checkpoint must provide all non-RW keys with matching "
            "shapes (change only RW-related flags such as walk length, wander_max_walk_len, wander_net, wander_net_n_heads, "
            "wander_layerscale, wander_pre_norm, scatter, tying). Incompatible with --resume_checkpoint."
        ),
    )
    ckpt.add_argument("--wandb_project", type=str, default="wander", help="Weights & Biases project name")
    ckpt.add_argument(
        "--checkpoint_metric",
        type=str,
        default="auto",
        choices=("auto", "accuracy", "roc_auc", "mrr"),
        help=(
            "Metric for best-checkpoint selection from the primary training "
            "eval split aggregate: auto (mean of available mrr+accuracy; "
            "legacy), or a single metric key (accuracy, roc_auc, mrr). "
            "Falls back to accuracy if the chosen metric is missing/NaN."
        ),
    )

    # ==================== Evaluation ====================
    evalg = parser.add_argument_group("evaluation")
    evalg.add_argument(
        "--test_datasets",
        type=str,
        nargs="+",
        default=["CORA"],
        help="Test dataset names (registry names only; optional NAME:version).",
    )
    evalg.add_argument(
        "--eval_batch_size_node",
        type=int,
        default=None,
        help=(
            "Node-classification batch size for val/test loaders. "
            "If unset (None, default), falls back to --batch_size_node."
        ),
    )
    evalg.add_argument(
        "--eval_batch_size_link",
        type=int,
        default=None,
        help=(
            "Link-prediction batch size for val/test loaders. "
            "If unset (None, default), falls back to --batch_size_link."
        ),
    )
    evalg.add_argument(
        "--wander_eval_walk_num",
        type=int,
        default=None,
        help=(
            "If set, overwrites --wander_walk_num during eval (val/test forwards). "
            "Still scaled by --wander_adaptive_walks. Default: --wander_walk_num."
        ),
    )
    evalg.add_argument(
        "--wander_keep_train_free_p",
        type=float,
        default=None,
        help=(
            "Eval Phase B (batch_only) only: keep walks that never visit a train "
            "node with this probability (0 = drop them). Train-hitting walks are "
            "always kept. Unset = no filter."
        ),
    )
    evalg.add_argument(
        "--wander_fast_uniform_walks",
        action="store_true",
        default=False,
        help=(
            "O(1) uniform neighbor sampling with cheap no-backtrack (rejection). "
            "Off by default (training uses the weighted-cumsum kernel). "
            "Eval scripts pass this flag; use --no_wander_fast_uniform_walks to force it off."
        ),
    )
    evalg.add_argument(
        "--no_wander_fast_uniform_walks",
        dest="wander_fast_uniform_walks",
        action="store_false",
        help="Use the original weighted-cumsum no-backtrack random walks.",
    )
    evalg.add_argument("--wander_test_samples", type=int, default=3, help="Number of test-time ensembling samples")
    evalg.add_argument(
        "--final_test_wander_samples",
        type=int,
        default=None,
        help=(
            "Optional override for wander_test_samples on the post-training "
            "best-checkpoint test eval only. Inline val/test eval during "
            "training still uses --wander_test_samples."
        ),
    )
    evalg.add_argument(
        "--wander_cached_train_kv",
        action="store_true",
        default=True,
        help=(
            "Node-classification eval only: two-phase cached train-KV eval. "
            "Phase A precomputes per-pass train-node hidden states once per "
            "dataset (stratified walks over train chunks of "
            "--wander_train_kv_chunk_size); Phase B evaluates test batches with "
            "batch-only walks, using the cached states as global-attention K/V. "
            "Phase A walk_num can be set independently via "
            "--wander_train_kv_walk_num. Requires --wander_global_attention_update "
            "and wander_n_layers == 1. With torchrun / world_size>1, Phase A runs "
            "on every rank (each needs the CPU cache); Phase B test batches are "
            "sharded round-robin across GPUs and metrics are all-reduced."
        ),
    )
    evalg.add_argument(
        "--no_wander_cached_train_kv",
        dest="wander_cached_train_kv",
        action="store_false",
        help="Disable cached train-KV eval (use per-batch train walks as global-attention K/V).",
    )
    evalg.add_argument(
        "--wander_train_kv_passes",
        type=int,
        default=16,
        help="Number of Phase A train-context passes for --wander_cached_train_kv.",
    )
    evalg.add_argument(
        "--final_test_train_kv_passes",
        type=int,
        default=None,
        help=(
            "Optional override for wander_train_kv_passes on the post-training "
            "best-checkpoint test eval only. Inline val/test eval during "
            "training still uses --wander_train_kv_passes."
        ),
    )
    evalg.add_argument(
        "--wander_train_kv_walk_num",
        type=int,
        default=None,
        help=(
            "Phase A only (--wander_cached_train_kv): base walk_num for train-context "
            "passes. Independent of --wander_walk_num, which controls training and "
            "(by default) Phase B test/query walks unless --wander_eval_walk_num is set. "
            "Default: --wander_walk_num."
        ),
    )
    evalg.add_argument(
        "--wander_train_kv_chunk_size",
        type=int,
        default=20000,
        help=(
            "Phase A only: stratified random-walk start pool size over train-node "
            "chunks for --wander_cached_train_kv. Independent of "
            "--batch_size_node / --eval_batch_size_node."
        ),
    )
    evalg.add_argument(
        "--wander_train_kv_cache_dtype",
        type=str,
        default="float32",
        help="CPU storage dtype for the cached train-KV hidden states (e.g. float32, bfloat16).",
    )
    evalg.add_argument(
        "--nc_proximity_batching",
        action="store_true",
        default=True,
        help=(
            "Node-classification eval only: pack eval batches from compact BFS "
            "balls around new max-degree uncovered seeds so batch nodes are "
            "graph-local. A full batch never continues the previous frontier; "
            "short balls are packed together until batch_size. Whole batches "
            "are sharded across DDP ranks. Unset --nc_proximity_batch_seed "
            "follows the experiment seed."
        ),
    )
    evalg.add_argument(
        "--no_nc_proximity_batching",
        dest="nc_proximity_batching",
        action="store_false",
        help="Disable proximity eval batching; sample eval nodes independently of graph locality.",
    )
    evalg.add_argument(
        "--nc_proximity_batch_max_radius",
        type=int,
        default=None,
        help=(
            "Max hop radius for --nc_proximity_batching. Unset means grow "
            "until batch_size or the component is exhausted. Requires "
            "--nc_proximity_batching."
        ),
    )
    evalg.add_argument(
        "--nc_proximity_batch_seed",
        type=int,
        default=None,
        help=(
            "RNG seed for --nc_proximity_batching (max-degree ties and "
            "last-layer overflow leftover, which stays uncovered). Unset: eval "
            "uses the experiment seed so "
            "--seeds 0 1 2 averages over different batch partitions. Set explicitly "
            "to pin a partition independently of the experiment seed."
        ),
    )
    evalg.add_argument(
        "--wander_compile_eval",
        action="store_true",
        default=False,
        help=(
            "torch.compile SwiGLU / GRU residual-FFN during eval (RMSNorm stays "
            "eager). Off during training (Wander.train disables it). "
            "Eval scripts turn this on; first forward compiles."
        ),
    )
    evalg.add_argument(
        "--no_wander_compile_eval",
        dest="wander_compile_eval",
        action="store_false",
        help="Disable eval torch.compile of SwiGLU / GRU residual-FFN.",
    )
    evalg.add_argument(
        "--evaluate_head_predictions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "During link-prediction evaluation (eval-only and per-epoch validation): "
            "if set, evaluate BOTH tail and head predictions for every test triple; "
            "if not set, evaluate only tail predictions. Head queries use "
            "inverse-relation queries with --add_inverse_edges_KGs, or predict_head=1 "
            "on the same triple with --no-add_inverse_edges_KGs. Default (unset) uses "
            "each dataset's preferred setting (True for KGs, False for AnyGraph)."
        ),
    )
    evalg.add_argument(
        "--link_pred_eval",
        type=str,
        choices=["mrr", "recall", "sampled_recall"],
        default=None,
        help=(
            "Link-prediction evaluation protocol. 'mrr': edge-based filtered ranking "
            "(filter all train+val+test positives) reporting MRR/Hits@K. 'recall': "
            "node-based AnyGraph-style protocol (rank candidates per source node, "
            "filter train positives only) reporting Recall@K/NDCG@K. "
            "'sampled_recall': per-source Recall@K on a cached pos ∪ sampled-neg "
            "list (shared Wander/UniLP comparison). Default (unset) uses each "
            "dataset's preferred protocol ('mrr' for KGs, 'recall' for AnyGraph, "
            "'sampled_recall' for UniLP)."
        ),
    )
    evalg.add_argument(
        "--recall_k",
        type=int,
        default=20,
        help="K for the --link_pred_eval recall protocol (AnyGraph default 20).",
    )
    evalg.add_argument(
        "--sampled_recall_num_neg",
        type=int,
        default=100,
        help="Sampled non-edges per source for --link_pred_eval sampled_recall.",
    )
    evalg.add_argument(
        "--sampled_recall_max_sources",
        type=int,
        default=None,
        help=(
            "If set, score this many sources per sampled-recall split "
            "(deterministic subset; same IDs as UniLP --max_sources)."
        ),
    )
    evalg.add_argument("--eval_only", action="store_true", help="Only evaluate (skip training)")
    evalg.add_argument(
        "--eval_batch_resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "With --eval_only, persist NC batch scores under "
            "<run>/eval_batch_resume/ so a preempted job can skip "
            "already-scored batches. Disable with --no-eval_batch_resume "
            "or FORCE_FRESH=1. Restart the same EVAL_RUN_DIR to resume."
        ),
    )
    evalg.add_argument(
        "--max_eval_samples_training",
        type=int,
        default=700,
        help=(
            "During training inline val / train-set eval: max test nodes (node "
            "classification) or queries (link prediction) per dataset per pass; "
            "subsample is fixed per dataset via seed."
        ),
    )
    evalg.add_argument(
        "--max_eval_samples_eval_only",
        type=int,
        default=None,
        help=(
            "With --eval_only: max test nodes (node classification) or queries "
            "(link prediction) per dataset; subsample is fixed per dataset via seed. "
            "None (default) means no cap."
        ),
    )
    evalg.add_argument(
        "--skip_train_eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Skip per-epoch inline evaluation on the train split during training "
            "(default: skip). Also skipped automatically when eval_batch_size_node "
            "is >= the number of train nodes (cached train-KV needs non-empty K/V "
            "after excluding query nodes). Val/test inline eval and the final "
            "post-training test eval are unchanged. Pass --no-skip_train_eval to "
            "enable train inline eval when compatible."
        ),
    )
    evalg.add_argument(
        "--skip_eval",
        action="store_true",
        help=(
            "Train and save epoch checkpoints without inline validation."
        ),
    )
    evalg.add_argument(
        "--skip_final_test",
        action="store_true",
        help=(
            "Skip the post-training best-checkpoint test eval (the expensive "
            "16-sample / 16-KV-pass pass). Inline val during training still "
            "runs, and model_seed*_best.pt is still saved."
        ),
    )

    return parser


def _dedupe_preserve_order(names: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _expand_dataset_list(names: list[str]) -> list[str]:
    if not names:
        return names
    return _dedupe_preserve_order(list(names))


def _dataset_registry_key(name: str) -> str:
    key = name.upper()
    if ":" in key:
        key, _ = key.split(":", 1)
    return key


def _require_registered_dataset_names(names: list[str], flag: str) -> list[str]:
    """Dedupe names and require each registry key to exist in DATASET_REGISTRY."""
    if not names:
        return names
    from data.datasets import DATASET_REGISTRY

    resolved = _expand_dataset_list(list(names))
    for name in resolved:
        key = _dataset_registry_key(name)
        if key not in DATASET_REGISTRY:
            raise ValueError(
                f"{flag} entry {name!r} is not a registered dataset name"
            )
    return resolved


def _resolve_dataset_name_list(
    names: list[str],
    *,
    allowed: Optional[list[str]] = None,
    flag: str,
) -> list[str]:
    """Dedupe names and optionally require membership in ``allowed``."""
    if not names:
        return []
    expanded = _expand_dataset_list(list(names))
    if allowed is None:
        return expanded
    allowed_by_upper = {n.upper(): n for n in allowed}
    resolved: list[str] = []
    for name in expanded:
        key = name.upper()
        if key not in allowed_by_upper:
            raise ValueError(
                f"{flag} entry {name!r} is not in the allowed set ({allowed})"
            )
        resolved.append(allowed_by_upper[key])
    return _dedupe_preserve_order(resolved)


def validate_args(args) -> None:
    """Validate argument combinations after parsing."""
    args.train_datasets = _require_registered_dataset_names(
        args.train_datasets, "--train_datasets"
    )
    args.test_datasets = _require_registered_dataset_names(
        args.test_datasets, "--test_datasets"
    )
    if args.wander_rotary_emb_intra_node_attention and not args.wander_intra_node_attention:
        raise ValueError(
            "--wander_rotary_emb_intra_node_attention requires --wander_intra_node_attention"
        )
    if args.wander_intra_node_ffn and not args.wander_intra_node_attention:
        raise ValueError(
            "--wander_intra_node_ffn requires --wander_intra_node_attention"
        )
    if args.wander_thinking_features < 0:
        raise ValueError("--wander_thinking_features must be >= 0")
    if args.wander_thinking_rows < 0:
        raise ValueError("--wander_thinking_rows must be >= 0")
    if args.wander_pre_norm and not args.wander_additive_refinement:
        raise ValueError(
            "--wander_pre_norm requires --wander_additive_refinement"
        )
    if (
        args.wander_global_kv_neighbors is not None
        and args.wander_global_kv_neighbors < 1
    ):
        raise ValueError("--wander_global_kv_neighbors must be >= 1 when set")
    if args.wander_cached_train_kv:
        if not args.wander_global_attention_update:
            raise ValueError(
                "--wander_cached_train_kv requires --wander_global_attention_update"
            )
        if args.wander_n_layers != 1:
            raise ValueError(
                "--wander_cached_train_kv requires --wander_n_layers 1 "
                "(K/V caching assumes a single global-attention layer)"
            )
        if args.wander_thinking_rows != 0:
            raise ValueError(
                "--wander_cached_train_kv does not support --wander_thinking_rows"
            )
        if args.wander_train_kv_passes < 1:
            raise ValueError("--wander_train_kv_passes must be >= 1")
        if args.wander_train_kv_chunk_size < 1:
            raise ValueError("--wander_train_kv_chunk_size must be >= 1")
        if (
            args.wander_train_kv_walk_num is not None
            and args.wander_train_kv_walk_num <= 0
        ):
            raise ValueError("--wander_train_kv_walk_num must be > 0")
    elif args.wander_train_kv_walk_num is not None:
        raise ValueError(
            "--wander_train_kv_walk_num requires --wander_cached_train_kv"
        )
    if (
        getattr(args, "final_test_train_kv_passes", None) is not None
        and args.final_test_train_kv_passes < 1
    ):
        raise ValueError("--final_test_train_kv_passes must be >= 1")
    if args.batch_size_node < 1 or args.batch_size_link < 1:
        raise ValueError("--batch_size_node and --batch_size_link must be >= 1")
    nc_query_frac = float(getattr(args, "nc_train_query_frac", 0.0) or 0.0)
    nc_query_cap = int(getattr(args, "nc_train_query_cap", 0) or 0)
    if not (0.0 <= nc_query_frac <= 1.0):
        raise ValueError("--nc_train_query_frac must be in [0, 1]")
    if nc_query_cap < 0:
        raise ValueError("--nc_train_query_cap must be >= 0")
    if getattr(args, "eval_batch_size_node", None) is not None and args.eval_batch_size_node < 1:
        raise ValueError("--eval_batch_size_node must be >= 1")
    max_r = getattr(args, "nc_proximity_batch_max_radius", None)
    if max_r is not None:
        if int(max_r) < 0:
            raise ValueError("--nc_proximity_batch_max_radius must be >= 0")
        if not getattr(args, "nc_proximity_batching", True):
            raise ValueError(
                "--nc_proximity_batch_max_radius requires --nc_proximity_batching"
            )
    if getattr(args, "nc_proximity_batch_seed", None) is not None:
        if not getattr(args, "nc_proximity_batching", True):
            raise ValueError(
                "--nc_proximity_batch_seed requires --nc_proximity_batching"
            )
    if getattr(args, "eval_batch_size_link", None) is not None and args.eval_batch_size_link < 1:
        raise ValueError("--eval_batch_size_link must be >= 1")
    if args.max_eval_samples_training < 1:
        raise ValueError("--max_eval_samples_training must be >= 1")
    if (
        args.max_eval_samples_eval_only is not None
        and args.max_eval_samples_eval_only < 1
    ):
        raise ValueError("--max_eval_samples_eval_only must be >= 1")
    if (
        getattr(args, "sampled_recall_max_sources", None) is not None
        and args.sampled_recall_max_sources < 1
    ):
        raise ValueError("--sampled_recall_max_sources must be >= 1")
    if args.num_graphs_batch < 1:
        raise ValueError("--num_graphs_batch must be >= 1")
    if not (0.0 <= args.synthetic_link_pred_prob <= 1.0):
        raise ValueError("--synthetic_link_pred_prob must be in [0, 1]")
    if args.synthetic_link_pred_prob > 0.0 and not args.synthetic_prior:
        raise ValueError(
            "--synthetic_link_pred_prob > 0 only takes effect together with --synthetic_prior"
        )
    if getattr(args, "prior_config", "default") != "default" and not args.synthetic_prior:
        raise ValueError("--prior_config only takes effect together with --synthetic_prior")
    if getattr(args, "features_first_only", False) and not args.synthetic_prior:
        raise ValueError(
            "--features_first_only only takes effect together with --synthetic_prior"
        )
    if getattr(args, "prior_max_node_cap", None) is not None:
        if not args.synthetic_prior:
            raise ValueError(
                "--prior_max_node_cap only takes effect together with --synthetic_prior"
            )
        if args.prior_max_node_cap < 2:
            raise ValueError("--prior_max_node_cap must be >= 2")
    if not (0.0 <= args.real_world_graph_prob <= 1.0):
        raise ValueError("--real_world_graph_prob must be in [0, 1]")
    if args.real_world_graph_prob > 0.0 and not args.synthetic_prior:
        raise ValueError(
            "--real_world_graph_prob > 0 only takes effect together with --synthetic_prior"
        )
    if args.real_world_graph_prob > 0.0 and not args.train_datasets:
        raise ValueError(
            "--real_world_graph_prob > 0 requires a non-empty --train_datasets"
        )
    if args.reset_rw_parameters:
        if not args.init_checkpoint:
            raise ValueError("--reset_rw_parameters requires --init_checkpoint")
        if args.resume_checkpoint:
            raise ValueError("--reset_rw_parameters cannot be used together with --resume_checkpoint")
    if getattr(args, "train_without_replacement", False) and not getattr(
        args, "node_cls_random_training_batches", True
    ):
        raise ValueError(
            "--train_without_replacement is not supported with class-balanced "
            "NC batches; keep --node_cls_random_training_batches (the default) "
            "or drop --train_without_replacement."
        )

    for name in (
        "pca_target_dim",
        "pca_target_dim_node_syn",
        "pca_target_dim_link_syn",
    ):
        if getattr(args, name, 32) < 0:
            raise ValueError(f"--{name} must be >= 0")
    if getattr(args, "pca_target_dim_link", None) is not None and args.pca_target_dim_link < 0:
        raise ValueError("--pca_target_dim_link must be >= 0")
    if getattr(args, "adaptive_pca_node_threshold", 0) < 0:
        raise ValueError("--adaptive_pca_node_threshold must be >= 0")

    if getattr(args, "skip_eval", False) and args.eval_only:
        raise ValueError("--skip_eval and --eval_only are mutually exclusive")
    if getattr(args, "skip_eval", False) and args.save_checkpoint_interval <= 0:
        raise ValueError("--skip_eval requires --save_checkpoint_interval > 0")
    if getattr(args, "patience", 0) < 0:
        raise ValueError("--patience must be >= 0")
    if getattr(args, "patience", 0) > 0 and getattr(args, "skip_eval", False):
        raise ValueError("--patience requires inline validation and cannot be used with --skip_eval")
    if args.ignore_features and args.dummy_features:
        raise ValueError("--ignore_features and --dummy_features cannot be used together")
    if args.wander_max_walk_len is None:
        args.wander_max_walk_len = args.wander_walk_len
    if args.wander_walk_num <= 0 or args.wander_walk_len <= 0 or args.wander_max_walk_len <= 0:
        raise ValueError("--wander_walk_num, --wander_walk_len, and --wander_max_walk_len must be > 0")
    if (
        getattr(args, "wander_eval_walk_num", None) is not None
        and args.wander_eval_walk_num <= 0
    ):
        raise ValueError("--wander_eval_walk_num must be > 0")
    keep_p = getattr(args, "wander_keep_train_free_p", None)
    if keep_p is not None and not (0.0 <= float(keep_p) <= 1.0):
        raise ValueError("--wander_keep_train_free_p must be in [0, 1]")
    if args.wander_walk_len > args.wander_max_walk_len:
        raise ValueError("--wander_walk_len must be <= --wander_max_walk_len")


def effective_eval_batch_size_node(args) -> int:
    """NC batch size for val/test loaders (falls back to ``batch_size_node``)."""
    v = getattr(args, "eval_batch_size_node", None)
    return int(args.batch_size_node if v is None else v)


def effective_eval_batch_size_link(args) -> int:
    """LP batch size for val/test loaders (falls back to ``batch_size_link``)."""
    v = getattr(args, "eval_batch_size_link", None)
    return int(args.batch_size_link if v is None else v)


def fill_missing_arg_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Backfill CLI defaults for keys missing from older ``training_args.json``."""
    parser = get_argument_parser()
    for action in parser._actions:
        dest = getattr(action, "dest", None)
        if not dest or dest == "help":
            continue
        if hasattr(args, dest):
            continue
        default = getattr(action, "default", None)
        if default is argparse.SUPPRESS:
            continue
        setattr(args, dest, default)
    return args


def prior_fixed_hp_overrides(args) -> dict:
    """Return keyword overrides for ``get_default_fixed_hp`` from CLI flags."""
    overrides = {}
    if getattr(args, "features_first_only", False) and args.synthetic_prior:
        overrides["features_first"] = True
    cap = getattr(args, "prior_max_node_cap", None)
    if cap is not None and args.synthetic_prior:
        # Match the default mixed_log_uniform template, only capping max_second.
        # Keep first/second split coherent when the cap is below the usual breakpoints.
        cap_f = float(cap)
        max_first = min(3000.0, cap_f)
        min_second = min(3000.0, cap_f)
        if min_second >= cap_f:
            min_second = max(200.0, cap_f * 0.5)
        if max_first >= cap_f:
            max_first = max(200.0, cap_f * 0.5)
        if max_first < 200.0:
            max_first = cap_f
        overrides["num_nodes"] = {
            "distribution": "mixed_log_uniform",
            "min_first": min(200.0, cap_f),
            "max_first": max_first,
            "min_second": min_second,
            "max_second": cap_f,
            "p_first": 0.5,
        }
    return overrides


def build_wander_config(args) -> dict:
    """Assemble a model_cfg dict for Wander from parsed arguments."""
    return {
        "test_samples": args.wander_test_samples,
        "use_ensemble_prefetch": getattr(args, "wander_use_ensemble_prefetch", True),
        "ensemble_prefetch_queue_depth": getattr(args, "wander_ensemble_prefetch_queue_depth", 2),
        "walk_num": args.wander_walk_num,
        "eval_walk_num": getattr(args, "wander_eval_walk_num", None),
        "keep_train_free_p": getattr(args, "wander_keep_train_free_p", None),
        "walk_len": args.wander_walk_len,
        "max_walk_len": args.wander_max_walk_len,
        "adaptive_walks": args.wander_adaptive_walks,
        "record_neighbors": args.wander_record_neighbors,
        "fast_uniform_walks": getattr(args, "wander_fast_uniform_walks", False),
        "compile_eval": getattr(args, "wander_compile_eval", False),
        "refinements": args.wander_refinements,
        "attention_scatter": args.wander_attention_scatter,
        "attention_scatter_n_heads": args.wander_attention_scatter_n_heads,
        "additive_refinement": args.wander_additive_refinement,
        "init_norm": args.wander_init_norm,
        "pre_norm": args.wander_pre_norm,
        "embed_only_first_refinement": args.wander_embed_first_only,
        "embedding_tying_across_refinements": args.wander_embedding_tying,
        "parameter_tying_across_refinements": args.wander_param_tying_across_refinements,
        "parameter_tying_across_channels": args.wander_param_tying_across_channels,
        "untie_attention_across_tasks": args.wander_untie_attention_across_tasks,
        "fixed_rw_across_refinements": args.wander_fixed_rw_across_refinements,
        "short_random_starts": getattr(args, "wander_short_random_starts", False),
        "hidden_dim": args.wander_hidden_dim,
        "dtype": args.wander_dtype,
        "net": args.wander_net,
        "n_layers": args.wander_n_layers,
        "net_n_heads": args.wander_net_n_heads,
        "intra_node_attention": args.wander_intra_node_attention,
        "intra_node_attention_n_heads": args.wander_intra_node_attention_n_heads,
        "rotary_emb_intra_node_attention": args.wander_rotary_emb_intra_node_attention,
        "intra_node_ffn": args.wander_intra_node_ffn,
        "inter_node_chunksize": args.wander_inter_node_chunksize,
        "intra_node_chunksize": args.wander_intra_node_chunksize,
        "checkpoint_refinements": args.wander_checkpoint_refinements,
        "checkpoint_sublayers": args.wander_checkpoint_sublayers,
        "layerscale": args.wander_layerscale,
        "global_attention_update": args.wander_global_attention_update,
        "global_kv_neighbors": args.wander_global_kv_neighbors,
        "rw_update": args.wander_rw_update,
        "disable_rw_graph_prune": args.wander_disable_rw_graph_prune,
        "feature_embedding_mlp": args.wander_feature_embedding_mlp,
        "feature_groups": args.wander_feature_groups,
        "concat_walks": args.wander_concat_walks,
        "randomize_feat_columns": args.wander_randomize_feat_columns,
        "randomize_label_columns": args.wander_randomize_label_columns,
        "nc_train_query_frac": float(getattr(args, "nc_train_query_frac", 0.0) or 0.0),
        "nc_train_query_cap": int(getattr(args, "nc_train_query_cap", 0) or 0),
        "cached_train_kv": args.wander_cached_train_kv,
        "train_kv_passes": args.wander_train_kv_passes,
        "train_kv_chunk_size": args.wander_train_kv_chunk_size,
        "train_kv_walk_num": (
            args.wander_walk_num
            if getattr(args, "wander_train_kv_walk_num", None) is None
            else int(args.wander_train_kv_walk_num)
        ),
        "train_kv_cache_dtype": args.wander_train_kv_cache_dtype,
        "thinking_features": args.wander_thinking_features,
        "thinking_rows": args.wander_thinking_rows,
        "add_inverse_edges_kgs": args.add_inverse_edges_KGs,
        "wander_only_forward_edges_when_inverses_added": args.wander_only_forward_edges_when_inverses_added,
    }
