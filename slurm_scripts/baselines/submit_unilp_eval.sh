#!/bin/bash
# Queue official UniLP (context_LP) inference.
#
# Usage (from the repo root, login node):
#   bash slurm_scripts/baselines/submit_unilp_eval.sh
#   UNILP_DATASETS=USAir,NS bash slurm_scripts/baselines/submit_unilp_eval.sh

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: do not sbatch this file; run it on the login node." >&2
  exit 1
fi

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
source "${_slurm}/_lib.sh"
cd "${REPO_ROOT}"
JOB_SCRIPT="${SLURM_SCRIPTS_DIR}/baselines/unilp_eval.sh"

mkdir -p logs

echo "Submitting official UniLP inference"
UNILP_ENV="${UNILP_ENV:-unilp_env}" \
  SLURM_JOB_NAME="${SLURM_JOB_NAME:-unilp-eval}" \
  bash slurm_scripts/submit.sh "${JOB_SCRIPT}"
