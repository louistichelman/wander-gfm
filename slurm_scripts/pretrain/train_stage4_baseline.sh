#!/usr/bin/env bash
#SBATCH --job-name=pt4-base
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm_pt4_%j.out
#SBATCH --error=logs/slurm_pt4_%j.err

# PT4-base: same recipe as PT3-base (3 KGs + synthetic prior, walks 128x128)
# but synthetic graphs are not capped by --prior_max_node_cap.
#
# Init from PT3-base epoch 99 (continue pretrain, no --reset_rw_parameters).
# Training uses --skip_eval.
#
# Submit from the repo root:
#   SLURM_PARTITION=gpu SLURM_NUM_GPUS=2 \
#     CLUSTER_MEM=256G \
#     bash slurm_scripts/submit.sh slurm_scripts/pretrain/train_stage4_baseline.sh
#
# First launch: --resume_checkpoint PT3 ep99.
# Later: auto-resumes latest checkpoints/<RUN_NAME>/model_seed*_epoch*.pt
# Override: FORCE_FRESH=1 | RESUME_CHECKPOINT=...

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

RUN_NAME="${RUN_NAME:-pt4}"
STAGE3_CKPT="${STAGE3_CKPT:-${CHECKPOINTS_DIR}/pt3/model_seed0_epoch99.pt}"
PRIOR_COMPLEXITY_START="${PRIOR_COMPLEXITY_START:-1.0}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

TRAIN_DATASETS=(FB15K_237 WN18RR CODEX_MEDIUM)
TEST_DATASETS=(CORA)

cluster_stage_datasets "${TRAIN_DATASETS[@]}" "${TEST_DATASETS[@]}"

CKPT_ARGS=()
if cluster_maybe_auto_resume "${RUN_NAME}" "PT4-base"; then
  CKPT_ARGS+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
else
  if [[ ! -f "${STAGE3_CKPT}" ]]; then
    echo "ERROR: stage-3 checkpoint not found: ${STAGE3_CKPT}" >&2
    exit 1
  fi
  STAGE3_CKPT="$(cluster_resolve_path "${STAGE3_CKPT}")"
  CKPT_ARGS+=(--resume_checkpoint "${STAGE3_CKPT}")
  echo "=== PT4-base: continue from ${STAGE3_CKPT} (uncapped num_nodes) ==="
fi

run_main \
  "${CKPT_ARGS[@]}" \
  --run_name "${RUN_NAME}" \
  --data_dir "${RAW_DATA_DIR}" \
  --save_load_path "${CHECKPOINTS_DIR}" \
  --batch_per_epoch 2000 \
  --synthetic_prior \
  --prior_complexity_start "${PRIOR_COMPLEXITY_START}" \
  --synthetic_link_pred_prob 0.2 \
  --train_datasets "${TRAIN_DATASETS[@]}" \
  --real_world_graph_prob 0.5 \
  --batch_size_link 4 \
  --test_datasets "${TEST_DATASETS[@]}" \
  --wander_walk_num 128 \
  --wander_walk_len 128 \
  --wander_max_walk_len 128 \
  --wander_adaptive_walks \
  --num_graphs_batch 2 \
  --max_epochs 150 \
  --batch_size_node 16 \
  --eval_batch_size_node 128 \
  --nc_proximity_batching \
  --wander_cached_train_kv \
  --wander_train_kv_passes 16 \
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
  --save_checkpoint_interval 1 \
  --add_inverse_edges_KGs \
  --skip_eval \
  --max_eval_samples_training 2000 \

finish_job
