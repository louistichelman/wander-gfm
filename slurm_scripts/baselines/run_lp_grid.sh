#!/usr/bin/env bash
#SBATCH --job-name=wander-lp-grid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/lp_grid_%x_%j.out
#SBATCH --error=logs/lp_grid_%x_%j.err

# One method × one dataset; runs the paper grid sequentially
# (BUDDY 12, NBFNet 18).
#   sbatch slurm_scripts/baselines/run_lp_grid.sh buddy ANYGRAPH_CORA
# Prefer queue_lp_grid.sh

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

METHOD="${1:-${METHOD:-buddy}}"
DATASETS="${2:-${DATASETS:-ANYGRAPH_CORA}}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_DIR}/baselines/lp_grid/${METHOD}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${CHECKPOINTS_DIR}/baselines/lp_grid/${METHOD}}"
DATA_DIR="${DATA_DIR:-${RAW_DATA_DIR}}"
EXTRA_CLI="${EXTRA_CLI:-}"
FORCE_FRESH="${FORCE_FRESH:-0}"

RESUME_FLAGS=()
if [[ "${FORCE_FRESH}" == "1" || "${FORCE_FRESH}" == "true" ]]; then
  RESUME_FLAGS+=(--no-resume)
fi

mkdir -p "${OUTPUT_DIR}" "${CHECKPOINT_DIR}"

echo "=== LP baseline grid ==="
echo "Method:      ${METHOD}"
echo "Datasets:    ${DATASETS}"
echo "Output:      ${OUTPUT_DIR}"
echo "Checkpoints: ${CHECKPOINT_DIR}"

"${PYTHON}" "${REPO_ROOT}/scripts/baselines/run_lp_baseline_grid.py" \
  --method "${METHOD}" \
  --datasets "${DATASETS}" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --device cuda \
  "${RESUME_FLAGS[@]}" \
  ${EXTRA_CLI}

echo "Results written to ${OUTPUT_DIR}"
finish_job
