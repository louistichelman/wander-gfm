#!/usr/bin/env bash
#SBATCH --job-name=pt1-graphpfn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm_pt1_ablation_graphpfn_prior_%j.out
#SBATCH --error=logs/slurm_pt1_ablation_graphpfn_prior_%j.err

# Ablation: PT1-base with --prior_config graphpfn (official GraphPFN prior).
# Same recipe as train_stage1_baseline.sh. NC skips GraphPFN regression
# tasks. LP (prob 0.2) reuses GraphPFN graphs as undirected label-stripped LP.
# Needs DGL in the Wander env (wander).
#
# Submit from the repo root:
#   SLURM_PARTITION=gpu SLURM_NUM_GPUS=2 CLUSTER_MEM=256G CLUSTER_CPUS=64 \
#     bash slurm_scripts/submit.sh slurm_scripts/ablations/train_stage1_ablation_graphpfn_prior.sh
#
# Auto-resumes from the latest checkpoints/<RUN_NAME>/model_seed*_epoch*.pt if present.
# Override: RESUME_CHECKPOINT=path | FORCE_FRESH=1 | RESUME_CHECKPOINT=none
# Optional: PRIOR_MAX_NODE_CAP (ignored by the GraphPFN prior; n_nodes is 1k-5k)
# Optional: SYNTHETIC_PREFETCH (default 48) SYNTHETIC_PREFETCH_WORKERS (default 24)
# GraphPFN sampling is Python/DGL-heavy (GIL). Workers are spawn processes kept
# alive across epochs. Throughput scales with workers until
# 2 ranks x workers ≈ CLUSTER_CPUS; beyond that they steal CPU from each other.
# Prefetch only buffers variance — it does not raise sample rate. Submit with
# CLUSTER_MEM=256G CLUSTER_CPUS=64 (ROMAN eval cache is ~18GB/rank).

set -euo pipefail

_slurm=""
if [[ -f "${SLURM_SUBMIT_DIR:-}/slurm_scripts/_lib.sh" ]]; then
  _slurm="${SLURM_SUBMIT_DIR}/slurm_scripts"
else
  _slurm="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  while [[ "${_slurm}" != "/" && ! -f "${_slurm}/_lib.sh" ]]; do
    _slurm="$(dirname "${_slurm}")"
  done
fi
# shellcheck source=/dev/null
source "${_slurm}/source_common.sh"

if ! "${PYTHON}" -c "import dgl" >/dev/null 2>&1; then
  echo "ERROR: prior_config=graphpfn needs DGL in ${CONDA_ENV} (${PYTHON})." >&2
  echo "Install a DGL wheel that matches this env's PyTorch/CUDA." >&2
  exit 1
fi

RUN_NAME="${RUN_NAME:-pt1_ablation_graphpfn_prior}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
PRIOR_MAX_NODE_CAP="${PRIOR_MAX_NODE_CAP:-8000}"
TEST_DATASETS=(CORA)

cluster_stage_datasets "${TEST_DATASETS[@]}"

RESUME_ARGS=()
if cluster_maybe_auto_resume "${RUN_NAME}" "PT1 ablation graphpfn prior"; then
  RESUME_ARGS+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
fi

echo "=== prior_config=graphpfn prior_max_node_cap=${PRIOR_MAX_NODE_CAP} prefetch=${SYNTHETIC_PREFETCH:-48} workers=${SYNTHETIC_PREFETCH_WORKERS:-24} mp=1 keep_workers=1 ==="

run_main \
  "${RESUME_ARGS[@]}" \
  --prior_config graphpfn \
  --prior_max_node_cap "${PRIOR_MAX_NODE_CAP}" \
  --run_name "${RUN_NAME}" \
  --data_dir "${RAW_DATA_DIR}" \
  --save_load_path "${CHECKPOINTS_DIR}" \
  --batch_per_epoch 2000 \
  --synthetic_prior \
  --prior_complexity_start 1.0 \
  --features_first_only \
  --synthetic_link_pred_prob 0.2 \
  --batch_size_link 4 \
  --test_datasets "${TEST_DATASETS[@]}" \
  --wander_walk_num 4 \
  --wander_walk_len 4 \
  --wander_max_walk_len 4 \
  --no_wander_rw_update \
  --wander_disable_rw_graph_prune \
  --num_graphs_batch 2 \
  --max_epochs 100 \
  --batch_size_node 9999 \
  --wander_inter_node_chunksize 128 \
  --wander_intra_node_chunksize 2048 \
  --node_cls_random_training_batches \
  --wander_global_attention_update \
  --pca_target_dim 16 \
  --pca_target_dim_link_syn 16 \
  --pca_target_dim_node_syn 16 \
  --no-pca_before_normalization \
  --final_inductive_zscore \
  --drop_constant_train_features \
  --save_checkpoint_interval 2 \
  --skip_eval \
  --max_eval_samples_training 10000 \
  --no_nc_proximity_batching \
  --synthetic_prefetch "${SYNTHETIC_PREFETCH:-48}" \
  --synthetic_prefetch_workers "${SYNTHETIC_PREFETCH_WORKERS:-24}" \
  --synthetic_prefetch_keep_workers \

finish_job
