#!/bin/bash
# Queue one GraphAny job per registry slug (raw tensors, Wander protocol splits).
#
# Usage (from the repo root, login node):
#   bash slurm_scripts/baselines/submit_graphany_eval.sh
#   DATASETS="cora citeseer" bash slurm_scripts/baselines/submit_graphany_eval.sh
#   GRAPHANY_NOFEAT=1 bash slurm_scripts/baselines/submit_graphany_eval.sh
#
# Prepare first (wander):
#   python baselines/graphany/prepare.py
#   python baselines/graphany/prepare.py --nofeat

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
SCRIPT_DIR="${SLURM_SCRIPTS_DIR}"
cd "${REPO_ROOT}"
JOB_SCRIPT="${SLURM_SCRIPTS_DIR}/baselines/graphany_eval.sh"
NOFEAT="${GRAPHANY_NOFEAT:-0}"

mkdir -p logs

if [[ -n "${DATASETS:-}" ]]; then
  # shellcheck disable=SC2206
  DATASET_LIST=(${DATASETS})
else
  SUBMIT_PYTHON="$(require_env_python "${WANDER_ENV:-wander}" "${SUBMIT_PYTHON:-}")"
  mapfile -t DATASET_LIST < <(
    PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${SUBMIT_PYTHON}" -c "
from baselines.registry import nc_eval_slugs
print('\n'.join(nc_eval_slugs(nofeat=${NOFEAT} == 1)))
"
  )
fi

echo "Submitting ${#DATASET_LIST[@]} GraphAny jobs (nofeat=${NOFEAT})"
for dataset in "${DATASET_LIST[@]}"; do
  job_name="graphany-${dataset}"
  echo "--- ${dataset} ---"
  GRAPHANY_DATASET="${dataset}" \
    GRAPHANY_NOFEAT="${NOFEAT}" \
    SLURM_JOB_NAME="${job_name}" \
    bash slurm_scripts/submit.sh "${JOB_SCRIPT}"
done
