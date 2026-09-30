#!/usr/bin/env bash
#SBATCH --job-name=grids-ptrue-plot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/slurm_eval_grids_p_true_plot_%x_%j.out
#SBATCH --error=logs/slurm_eval_grids_p_true_plot_%x_%j.err

# Rebuild Grids p(true class) panel figures from collected maps.
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

PLOT_SCRIPT="${REPO_ROOT}/scripts/eval/plot_grids_p_true_maps.py"
SWEEP_DIR="${SWEEP_DIR:-${HOME_RESULTS_DIR}/eval/grids_p_true_maps}"
if [[ "${SWEEP_DIR}" == "${HOME_RESULTS_DIR}"/* ]]; then
  SWEEP_DIR="${RESULTS_DIR}/${SWEEP_DIR#${HOME_RESULTS_DIR}/}"
fi

cd "${REPO_ROOT}"
"$PYTHON" "${PLOT_SCRIPT}" --sweep_dir "${SWEEP_DIR}"
echo "Summary: ${SWEEP_DIR}/grids_p_true_maps_summary.txt"
finish_job
