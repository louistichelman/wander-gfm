#!/usr/bin/env bash
#SBATCH --job-name=pt3-nodepfn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm_pt3_ablation_nodepfn_prior_%j.out
#SBATCH --error=logs/slurm_pt3_ablation_nodepfn_prior_%j.err

# Ablation: PT3-base with --prior_config nodepfn (official NodePFN prior).
# Same recipe as train_stage3_baseline.sh: init from PT2 epoch 199
# with --reset_rw_parameters, walks 128 adaptive, real+syn mix, PCA 64,
# GraphLand preprocess, skip_eval. Needs gpytorch + ConfigSpace.
#
# Submit from the repo root:
#   SLURM_PARTITION=gpu SLURM_NUM_GPUS=2 \
#     PRIOR_MAX_NODE_CAP=8000 CLUSTER_MEM=256G \
#     bash slurm_scripts/submit.sh slurm_scripts/ablations/train_stage3_ablation_nodepfn_prior.sh
#
# First launch: --init_checkpoint PT2 ep199 + --reset_rw_parameters.
# Later: auto-resumes latest checkpoints/<RUN_NAME>/model_seed*_epoch*.pt
#   (no reset_rw on resume). Override: FORCE_FRESH=1 | RESUME_CHECKPOINT=...

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

if ! "${PYTHON}" -c "import gpytorch, ConfigSpace" >/dev/null 2>&1; then
  echo "ERROR: prior_config=nodepfn needs gpytorch and ConfigSpace in ${CONDA_ENV} (${PYTHON})." >&2
  echo "Fix: ${PYTHON} -m pip install -r requirements-priors.txt" >&2
  exit 1
fi

RUN_NAME="${RUN_NAME:-pt3_ablation_nodepfn_prior}"
STAGE2_CKPT="${CHECKPOINTS_DIR}/pt2_ablation_nodepfn_prior/model_seed0_epoch199.pt"
# Paper training uses prior_complexity=1.0 for the whole run.
# Official NodePFN prior ignores complexity; keep the value for logging / adapt.
PRIOR_COMPLEXITY_START="${PRIOR_COMPLEXITY_START:-1.0}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

TRAIN_DATASETS=(FB15K_237 WN18RR CODEX_MEDIUM)
TEST_DATASETS=(CORA)

cluster_stage_datasets "${TRAIN_DATASETS[@]}" "${TEST_DATASETS[@]}"

CKPT_ARGS=()
if cluster_maybe_auto_resume "${RUN_NAME}" "PT3 ablation nodepfn prior"; then
  CKPT_ARGS+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
else
  if [[ ! -f "${STAGE2_CKPT}" ]]; then
    echo "ERROR: stage-2 checkpoint not found: ${STAGE2_CKPT}" >&2
    exit 1
  fi
  STAGE2_CKPT="$(cluster_resolve_path "${STAGE2_CKPT}")"
  CKPT_ARGS+=(--init_checkpoint "${STAGE2_CKPT}" --reset_rw_parameters)
  echo "=== PT3 ablation nodepfn prior: init from ${STAGE2_CKPT} (--reset_rw_parameters) ==="
fi

PRIOR_CAP_ARGS=()
if [[ -n "${PRIOR_MAX_NODE_CAP:-}" ]]; then
  PRIOR_CAP_ARGS+=(--prior_max_node_cap "${PRIOR_MAX_NODE_CAP}")
  echo "=== prior_max_node_cap=${PRIOR_MAX_NODE_CAP} ==="
fi

echo "=== prior_config=nodepfn prior_complexity_start=${PRIOR_COMPLEXITY_START} ==="

run_main \
  "${CKPT_ARGS[@]}" \
  "${PRIOR_CAP_ARGS[@]}" \
  --prior_config nodepfn \
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
  --max_epochs 100 \
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
  --save_checkpoint_interval 2 \
  --add_inverse_edges_KGs \
  --skip_eval \
  --max_eval_samples_training 2000 \

finish_job
