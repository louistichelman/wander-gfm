#!/usr/bin/env bash
#SBATCH --job-name=wander-heur
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/heuristics_%x_%j.out
#SBATCH --error=logs/heuristics_%x_%j.err

# CN / RA / PA eval for one homogeneous AnyGraph LP dataset (CPU).
# Submit via queue_lp_heuristics.sh, or:
#   sbatch slurm_scripts/baselines/run_lp_heuristics.sh ANYGRAPH_CORA

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

DATASETS="${1:-${DATASETS:-ANYGRAPH_CORA}}"
SEEDS="${SEEDS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_DIR}/baselines/heuristics}"
DATA_DIR="${DATA_DIR:-${RAW_DATA_DIR}}"
EXTRA_CLI="${EXTRA_CLI:-}"

mkdir -p "${OUTPUT_DIR}"

echo "=== LP heuristics (CN / RA / PA) ==="
echo "Datasets: ${DATASETS}"
echo "Seeds:    ${SEEDS}"
echo "Output:   ${OUTPUT_DIR}"
echo "Data:     ${DATA_DIR}"

# Heuristics are CPU-only; hide any allocated GPU.
export CUDA_VISIBLE_DEVICES=""

"${PYTHON}" "${REPO_ROOT}/scripts/baselines/run_lp_heuristics.py" \
  --datasets "${DATASETS}" \
  --seeds "${SEEDS}" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  ${EXTRA_CLI}

echo "Results written to ${OUTPUT_DIR}"
finish_job
