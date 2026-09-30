#!/usr/bin/env bash
#SBATCH --job-name=pt1-no-syn-lp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm_pt1_ablation_syn_link_pred_0_%j.out
#SBATCH --error=logs/slurm_pt1_ablation_syn_link_pred_0_%j.err

# Ablation: PT1-base with synthetic_link_pred_prob=0 (node-cls only on synthetic prior).
# Otherwise identical to train_stage1_baseline.sh.
#
# Submit from the repo root:
#   SLURM_PARTITION=gpu SLURM_NUM_GPUS=2 \
#     bash slurm_scripts/submit.sh slurm_scripts/train_stage1_ablation_syn_link_pred_0.sh
#
# Auto-resumes from the latest checkpoints/<RUN_NAME>/model_seed*_epoch*.pt if present.
# Override: RESUME_CHECKPOINT=path | FORCE_FRESH=1 | RESUME_CHECKPOINT=none
# Optional: PRIOR_MAX_NODE_CAP (default 8000)

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

RUN_NAME="${RUN_NAME:-pt1_ablation_syn_link_pred_0}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
PRIOR_MAX_NODE_CAP="${PRIOR_MAX_NODE_CAP:-8000}"
TEST_DATASETS=(CORA)

cluster_stage_datasets "${TEST_DATASETS[@]}"

RESUME_ARGS=()
if cluster_maybe_auto_resume "${RUN_NAME}" "PT1 ablation syn_link_pred=0"; then
  RESUME_ARGS+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
fi

echo "=== prior_max_node_cap=${PRIOR_MAX_NODE_CAP} ==="

run_main \
  "${RESUME_ARGS[@]}" \
  --prior_max_node_cap "${PRIOR_MAX_NODE_CAP}" \
  --run_name "${RUN_NAME}" \
  --data_dir "${RAW_DATA_DIR}" \
  --save_load_path "${CHECKPOINTS_DIR}" \
  --batch_per_epoch 2000 \
  --synthetic_prior \
  --prior_complexity_start 1.0 \
  --features_first_only \
  --synthetic_link_pred_prob 0 \
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
  --max_eval_samples_eval_only 1000 \
  --no_nc_proximity_batching \
  --no_wander_cached_train_kv \

finish_job
