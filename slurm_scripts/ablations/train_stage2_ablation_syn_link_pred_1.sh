#!/usr/bin/env bash
#SBATCH --job-name=pt2-syn-lp1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm_pt2_ablation_syn_link_pred_1_%j.out
#SBATCH --error=logs/slurm_pt2_ablation_syn_link_pred_1_%j.err

# Ablation: PT2-style continuation of syn_link_pred=1 stage-1.
# Resume pt1_ablation_syn_link_pred_1 epoch19, PCA 32, 20 more epochs
# (max_epochs=40 → epochs 20-39), synthetic LP only.
#
# Submit from the repo root:
#   SLURM_PARTITION=gpu SLURM_NUM_GPUS=2 \
#     bash slurm_scripts/submit.sh slurm_scripts/ablations/train_stage2_ablation_syn_link_pred_1.sh
#
# Prefer continuing this run's own latest ckpt if already started; else stage-1.
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

RUN_NAME="${RUN_NAME:-pt2_ablation_syn_link_pred_1}"
STAGE1_CKPT="${CHECKPOINTS_DIR}/pt1_ablation_syn_link_pred_1/model_seed0_epoch19.pt"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
PRIOR_MAX_NODE_CAP="${PRIOR_MAX_NODE_CAP:-8000}"
TEST_DATASETS=(CORA)

cluster_stage_datasets "${TEST_DATASETS[@]}"

RESUME_ARGS=()
if cluster_maybe_auto_resume "${RUN_NAME}" "PT2 ablation syn_link_pred=1"; then
  RESUME_ARGS+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
elif [[ -z "${RESUME_CHECKPOINT}" && -f "${STAGE1_CKPT}" ]]; then
  RESUME_CHECKPOINT="${STAGE1_CKPT}"
  RESUME_ARGS+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
  echo "=== PT2 ablation syn_link_pred=1: resuming from stage-1 ${RESUME_CHECKPOINT} ==="
elif [[ -n "${RESUME_CHECKPOINT}" && "${RESUME_CHECKPOINT}" != "none" ]]; then
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
  --synthetic_link_pred_prob 1 \
  --batch_size_link 4 \
  --test_datasets "${TEST_DATASETS[@]}" \
  --wander_walk_num 4 \
  --wander_walk_len 4 \
  --wander_max_walk_len 4 \
  --no_wander_rw_update \
  --wander_disable_rw_graph_prune \
  --num_graphs_batch 2 \
  --max_epochs 40 \
  --batch_size_node 9999 \
  --wander_inter_node_chunksize 128 \
  --wander_intra_node_chunksize 2048 \
  --node_cls_random_training_batches \
  --wander_global_attention_update \
  --pca_target_dim 32 \
  --pca_target_dim_link_syn 32 \
  --pca_target_dim_node_syn 32 \
  --no-pca_before_normalization \
  --final_inductive_zscore \
  --graphland_categorical_as_ordinals \
  --graphland_different_transform \
  --drop_constant_train_features \
  --save_checkpoint_interval 2 \
  --skip_eval \
  --max_eval_samples_training 10000 \
  --max_eval_samples_eval_only 1000 \
  --no_nc_proximity_batching \
  --no_wander_cached_train_kv \

finish_job
