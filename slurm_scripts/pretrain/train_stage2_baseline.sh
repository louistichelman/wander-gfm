#!/usr/bin/env bash
#SBATCH --job-name=pt2-base
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm_pt2_baseline_%j.out
#SBATCH --error=logs/slurm_pt2_baseline_%j.err

# Experiment plan: PT2-base (stage 2, epochs 100-199).
# Resume PT1-base (pt1 epoch99), PCA 32 for link+node,
# same prior; complexity stays at 1.0 (restored from ckpt if resuming).
# GraphLand eval preprocess: categorical_as_ordinals, different_transform,
# drop_constant_train_features.
#
# Submit from the repo root:
#   SLURM_PARTITION=gpu SLURM_NUM_GPUS=2 \
#     PRIOR_MAX_NODE_CAP=8000 \
#     bash slurm_scripts/submit.sh slurm_scripts/pretrain/train_stage2_baseline.sh
#
# Default resume: checkpoints/pt1/model_seed0_epoch99.pt
# Override: RESUME_CHECKPOINT=path | FORCE_FRESH=1 | RESUME_CHECKPOINT=none
# Optional: PRIOR_MAX_NODE_CAP=8000
# Training uses --skip_eval.

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

RUN_NAME="${RUN_NAME:-pt2}"
STAGE1_CKPT="${CHECKPOINTS_DIR}/pt1/model_seed0_epoch99.pt"
# Prefer continuing this run's own latest ckpt if already started; else stage-1.
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
TEST_DATASETS=(CORA)

cluster_stage_datasets "${TEST_DATASETS[@]}"

RESUME_ARGS=()
if cluster_maybe_auto_resume "${RUN_NAME}" "PT2-base"; then
  RESUME_ARGS+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
elif [[ -z "${RESUME_CHECKPOINT}" && -f "${STAGE1_CKPT}" ]]; then
  RESUME_CHECKPOINT="${STAGE1_CKPT}"
  RESUME_ARGS+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
  echo "=== PT2-base: resuming from stage-1 ${RESUME_CHECKPOINT} ==="
elif [[ -n "${RESUME_CHECKPOINT}" && "${RESUME_CHECKPOINT}" != "none" ]]; then
  RESUME_ARGS+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
fi

PRIOR_CAP_ARGS=()
if [[ -n "${PRIOR_MAX_NODE_CAP:-}" ]]; then
  PRIOR_CAP_ARGS+=(--prior_max_node_cap "${PRIOR_MAX_NODE_CAP}")
  echo "=== prior_max_node_cap=${PRIOR_MAX_NODE_CAP} ==="
fi

run_main \
  "${RESUME_ARGS[@]}" \
  "${PRIOR_CAP_ARGS[@]}" \
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
  --max_epochs 200 \
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

finish_job
