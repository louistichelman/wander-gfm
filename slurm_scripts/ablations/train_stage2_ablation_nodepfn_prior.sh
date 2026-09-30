#!/usr/bin/env bash
#SBATCH --job-name=pt2-nodepfn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm_pt2_ablation_nodepfn_prior_%j.out
#SBATCH --error=logs/slurm_pt2_ablation_nodepfn_prior_%j.err

# Ablation: PT2-base with --prior_config nodepfn (official NodePFN prior).
# Same recipe as train_stage2_baseline.sh: resume PT1 epoch 99,
# PCA 32, max_epochs 200 (epochs 100-199), GraphLand eval preprocess.
# Needs gpytorch + ConfigSpace in wander.
#
# Submit from the repo root:
#   SLURM_PARTITION=gpu SLURM_NUM_GPUS=2 CLUSTER_MEM=128G \
#     bash slurm_scripts/submit.sh slurm_scripts/ablations/train_stage2_ablation_nodepfn_prior.sh
#
# Default resume: checkpoints/pt1_ablation_nodepfn_prior/model_seed0_epoch99.pt
# Prefer continuing this run's own latest ckpt if already started; else stage-1.
# Override: RESUME_CHECKPOINT=path | FORCE_FRESH=1 | RESUME_CHECKPOINT=none
# Optional: PRIOR_MAX_NODE_CAP (ignored by the NodePFN prior; seq_len is 1024)

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

RUN_NAME="${RUN_NAME:-pt2_ablation_nodepfn_prior}"
STAGE1_CKPT="${CHECKPOINTS_DIR}/pt1_ablation_nodepfn_prior/model_seed0_epoch99.pt"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
PRIOR_MAX_NODE_CAP="${PRIOR_MAX_NODE_CAP:-8000}"
TEST_DATASETS=(CORA)

cluster_stage_datasets "${TEST_DATASETS[@]}"

RESUME_ARGS=()
if cluster_maybe_auto_resume "${RUN_NAME}" "PT2 ablation nodepfn prior"; then
  RESUME_ARGS+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
elif [[ -z "${RESUME_CHECKPOINT}" && -f "${STAGE1_CKPT}" ]]; then
  RESUME_CHECKPOINT="${STAGE1_CKPT}"
  RESUME_ARGS+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
  echo "=== PT2 ablation nodepfn prior: resuming from stage-1 ${RESUME_CHECKPOINT} ==="
elif [[ -n "${RESUME_CHECKPOINT}" && "${RESUME_CHECKPOINT}" != "none" ]]; then
  RESUME_ARGS+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
fi

echo "=== prior_config=nodepfn prior_max_node_cap=${PRIOR_MAX_NODE_CAP} ==="

run_main \
  "${RESUME_ARGS[@]}" \
  --prior_config nodepfn \
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
  --no_nc_proximity_batching \

finish_job
