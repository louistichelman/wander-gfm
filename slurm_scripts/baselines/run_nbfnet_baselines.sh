#!/usr/bin/env bash
#SBATCH --job-name=nbfnet-lp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm_nbfnet_baselines_%j.out
#SBATCH --error=logs/slurm_nbfnet_baselines_%j.err

# Train/eval NBFNet LP baseline on AnyGraph LP datasets.
#
# Usage (from the repo root):
#   bash slurm_scripts/submit.sh slurm_scripts/baselines/run_nbfnet_baselines.sh
#   DATASETS=ANYGRAPH_CORA SEEDS=0 bash slurm_scripts/submit.sh slurm_scripts/baselines/run_nbfnet_baselines.sh
#   DATASETS="ANYGRAPH_CORA ANYGRAPH_DDI" bash slurm_scripts/submit.sh slurm_scripts/baselines/run_nbfnet_baselines.sh
#
# Resume:
#   - Trainer auto-resumes from ${CHECKPOINT_DIR}/{DATASET}_seed{N}_best.pt when present
#     (best weights + last_completed_epoch). Set FORCE_FRESH=1 to delete ckpts and
#     start over, or pass EXTRA_CLI='--no-resume'.
#   - Finished runs with a success JSON are skipped (EXTRA_CLI='--no-skip_finished' to redo).
#
# Outputs: results/baselines/nbfnet_*/{DATASET}_seed{N}.json
# Checkpoints: checkpoints/baselines/nbfnet/{DATASET}_seed{N}_best.pt

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

DATASETS="${DATASETS:-ANYGRAPH_CORA}"
SEEDS="${SEEDS:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
MAX_EVAL_QUERIES="${MAX_EVAL_QUERIES:-}"
USE_NODE_FEATURES="${USE_NODE_FEATURES:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_DIR}/baselines/nbfnet_$(date +%y%m%d_%H%M)}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${CHECKPOINTS_DIR}/baselines/nbfnet}"
EXTRA_CLI="${EXTRA_CLI:-}"
FORCE_FRESH="${FORCE_FRESH:-0}"

FEAT_FLAG="--use_node_features"
if [[ "${USE_NODE_FEATURES}" == "0" || "${USE_NODE_FEATURES}" == "false" ]]; then
  FEAT_FLAG="--no-use_node_features"
fi

RESUME_FLAGS=()
if [[ "${FORCE_FRESH}" == "1" || "${FORCE_FRESH}" == "true" ]]; then
  RESUME_FLAGS+=(--no-resume)
  echo "=== FORCE_FRESH=1: will ignore/remove existing *_best.pt ==="
fi

EVAL_CAP_FLAGS=()
if [[ -n "${MAX_EVAL_QUERIES}" ]]; then
  EVAL_CAP_FLAGS+=(--max_eval_queries "${MAX_EVAL_QUERIES}")
fi

mkdir -p logs "${OUTPUT_DIR}" "${CHECKPOINT_DIR}"

echo "=== NBFNet LP baseline ==="
echo "Datasets:    ${DATASETS}"
echo "Seeds:       ${SEEDS}"
echo "Output:      ${OUTPUT_DIR}"
echo "Checkpoints: ${CHECKPOINT_DIR}"
echo "Features: ${USE_NODE_FEATURES}"
echo "Slurm job: ${SLURM_JOB_ID:-local} requeue=${SLURM_RESTART_COUNT:-0}"
if [[ -n "${SLURM_RESTART_COUNT:-}" && "${SLURM_RESTART_COUNT}" != "0" ]]; then
  echo "=== Requeued after preemption (restart #${SLURM_RESTART_COUNT}); trainer will resume from *_best.pt if present ==="
fi

cd "${REPO_ROOT}"
"${PYTHON}" "${REPO_ROOT}/scripts/baselines/run_nbfnet_baselines.py" \
  --datasets "${DATASETS}" \
  --seeds "${SEEDS}" \
  --data_dir "${DATA_DIR:-./raw_data}" \
  --output_dir "${OUTPUT_DIR}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --max_epochs "${MAX_EPOCHS}" \
  "${EVAL_CAP_FLAGS[@]}" \
  ${FEAT_FLAG} \
  "${RESUME_FLAGS[@]}" \
  ${EXTRA_CLI}

echo "Results written to ${OUTPUT_DIR}"
finish_job
