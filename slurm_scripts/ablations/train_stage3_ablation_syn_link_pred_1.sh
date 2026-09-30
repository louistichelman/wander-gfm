#!/usr/bin/env bash
#SBATCH --job-name=pt3-syn-lp1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm_pt3_ablation_syn_link_pred_1_%j.out
#SBATCH --error=logs/slurm_pt3_ablation_syn_link_pred_1_%j.err

# Ablation: PT3-style (full RW) continuation of syn_link_pred=1.
# Init from pt2_ablation_syn_link_pred_1 epoch39 with --reset_rw_parameters.
# Synthetic LP only, prior_complexity_start=1.0, 20 epochs, walks 128 adaptive.
#
# Training uses --skip_eval.
#
# Submit from the repo root:
#   SLURM_PARTITION=gpu SLURM_NUM_GPUS=2 \
#     bash slurm_scripts/submit.sh slurm_scripts/ablations/train_stage3_ablation_syn_link_pred_1.sh
#
# First launch: --init_checkpoint PT2 ep39 + --reset_rw_parameters.
# Later: auto-resumes latest checkpoints/<RUN_NAME>/model_seed*_epoch*.pt
#   (no reset_rw on resume). Override: FORCE_FRESH=1 | RESUME_CHECKPOINT=...
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

RUN_NAME="${RUN_NAME:-pt3_ablation_syn_link_pred_1}"
STAGE2_CKPT="${CHECKPOINTS_DIR}/pt2_ablation_syn_link_pred_1/model_seed0_epoch39.pt"
PRIOR_COMPLEXITY_START="${PRIOR_COMPLEXITY_START:-1.0}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
PRIOR_MAX_NODE_CAP="${PRIOR_MAX_NODE_CAP:-8000}"
TEST_DATASETS=(CORA)

cluster_stage_datasets "${TEST_DATASETS[@]}"

CKPT_ARGS=()
if cluster_maybe_auto_resume "${RUN_NAME}" "PT3 ablation syn_link_pred=1"; then
  CKPT_ARGS+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
else
  if [[ ! -f "${STAGE2_CKPT}" ]]; then
    echo "ERROR: stage-2 checkpoint not found: ${STAGE2_CKPT}" >&2
    exit 1
  fi
  STAGE2_CKPT="$(cluster_resolve_path "${STAGE2_CKPT}")"
  CKPT_ARGS+=(--init_checkpoint "${STAGE2_CKPT}" --reset_rw_parameters)
  echo "=== PT3 ablation syn_link_pred=1: init from ${STAGE2_CKPT} (--reset_rw_parameters) ==="
fi

echo "=== prior_max_node_cap=${PRIOR_MAX_NODE_CAP} prior_complexity_start=${PRIOR_COMPLEXITY_START} ==="

run_main \
  "${CKPT_ARGS[@]}" \
  --prior_max_node_cap "${PRIOR_MAX_NODE_CAP}" \
  --run_name "${RUN_NAME}" \
  --data_dir "${RAW_DATA_DIR}" \
  --save_load_path "${CHECKPOINTS_DIR}" \
  --batch_per_epoch 2000 \
  --synthetic_prior \
  --prior_complexity_start "${PRIOR_COMPLEXITY_START}" \
  --synthetic_link_pred_prob 1 \
  --batch_size_link 4 \
  --test_datasets "${TEST_DATASETS[@]}" \
  --wander_walk_num 128 \
  --wander_walk_len 128 \
  --wander_max_walk_len 128 \
  --wander_adaptive_walks \
  --num_graphs_batch 2 \
  --max_epochs 20 \
  --batch_size_node 16 \
  --wander_inter_node_chunksize 128 \
  --wander_intra_node_chunksize 2048 \
  --node_cls_random_training_batches \
  --wander_global_attention_update \
  --pca_target_dim 64 \
  --pca_target_dim_link_syn 32 \
  --pca_target_dim_node_syn 64 \
  --no-pca_before_normalization \
  --final_inductive_zscore \
  --graphland_categorical_as_ordinals \
  --graphland_different_transform \
  --drop_constant_train_features \
  --save_checkpoint_interval 2 \
  --max_eval_samples_training 2000 \
  --max_eval_samples_eval_only 1000 \
  --no_nc_proximity_batching \
  --no_wander_cached_train_kv \
  --skip_eval \

finish_job
