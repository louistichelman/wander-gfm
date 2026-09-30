#!/usr/bin/env bash
#SBATCH --job-name=pt3-ablate-syn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm_pt3_ablate_syn_only_%j.out
#SBATCH --error=logs/slurm_pt3_ablate_syn_only_%j.err

# PT3 synthetic-only ablation (no real-world train graphs).
# Resume latest checkpoints/<RUN_NAME>/model_seed*_epoch*.pt by default.
#
# Training uses --skip_eval.
#
# Submit from the repo root:
#   SLURM_PARTITION=gpu SLURM_NUM_GPUS=2 CLUSTER_MEM=256G \
#     PRIOR_MAX_NODE_CAP=8000 \
#     SAVE_CHECKPOINT_INTERVAL=1 \
#     bash slurm_scripts/submit.sh slurm_scripts/ablations/train_stage3_ablation_syn_only.sh

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

RUN_NAME="${RUN_NAME:-pt3_ablation_syn_only}"
PRIOR_COMPLEXITY_START="${PRIOR_COMPLEXITY_START:-1.0}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
REAL_WORLD_GRAPH_PROB="${REAL_WORLD_GRAPH_PROB:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
SAVE_CHECKPOINT_INTERVAL="${SAVE_CHECKPOINT_INTERVAL:-1}"
PRIOR_MAX_NODE_CAP="${PRIOR_MAX_NODE_CAP:-8000}"
TEST_DATASETS=(CORA)

cluster_stage_datasets "${TEST_DATASETS[@]}"

CKPT_ARGS=()
if cluster_maybe_auto_resume "${RUN_NAME}" "PT3-ablate-syn-only"; then
  CKPT_ARGS+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
else
  echo "ERROR: no checkpoint to resume for ${RUN_NAME}" >&2
  exit 1
fi

PRIOR_CAP_ARGS=()
if [[ -n "${PRIOR_MAX_NODE_CAP}" ]]; then
  PRIOR_CAP_ARGS+=(--prior_max_node_cap "${PRIOR_MAX_NODE_CAP}")
fi

echo "=== prior_max_node_cap=${PRIOR_MAX_NODE_CAP:-<none>} ==="
echo "=== real_world_graph_prob=${REAL_WORLD_GRAPH_PROB} test_datasets=${TEST_DATASETS[*]} max_epochs=${MAX_EPOCHS} save_interval=${SAVE_CHECKPOINT_INTERVAL} ==="

run_main \
  "${CKPT_ARGS[@]}" \
  "${PRIOR_CAP_ARGS[@]}" \
  --run_name "${RUN_NAME}" \
  --data_dir "${RAW_DATA_DIR}" \
  --save_load_path "${CHECKPOINTS_DIR}" \
  --batch_per_epoch 2000 \
  --synthetic_prior \
  --prior_complexity_start "${PRIOR_COMPLEXITY_START}" \
  --synthetic_link_pred_prob 0.2 \
  --real_world_graph_prob "${REAL_WORLD_GRAPH_PROB}" \
  --batch_size_link 4 \
  --test_datasets "${TEST_DATASETS[@]}" \
  --wander_walk_num 128 \
  --wander_walk_len 128 \
  --wander_max_walk_len 128 \
  --wander_adaptive_walks \
  --num_graphs_batch 2 \
  --max_epochs "${MAX_EPOCHS}" \
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
  --save_checkpoint_interval "${SAVE_CHECKPOINT_INTERVAL}" \
  --no-add_inverse_edges_KGs \
  --skip_eval \
  --max_eval_samples_training 2000 \

finish_job
