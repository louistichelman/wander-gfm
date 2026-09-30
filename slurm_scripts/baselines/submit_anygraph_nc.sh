#!/bin/bash
# Queue one AnyGraph NC job per Wander NC-eval slug.
#
# Usage (from the repo root, login node):
#   bash slurm_scripts/baselines/submit_anygraph_nc.sh
#   ANYGRAPH_NOFEAT=1 bash slurm_scripts/baselines/submit_anygraph_nc.sh

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
JOB_SCRIPT="${SCRIPT_DIR}/baselines/anygraph_nc.sh"
ANYGRAPH_NOFEAT="${ANYGRAPH_NOFEAT:-0}"

mkdir -p logs

SUBMIT_PYTHON="$(require_env_python "${WANDER_ENV:-wander}" "${SUBMIT_PYTHON:-}")"
mapfile -t DATASETS < <(
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${SUBMIT_PYTHON}" -c "
from baselines.registry import nc_eval_slugs
nofeat = '${ANYGRAPH_NOFEAT}' == '1'
print('\n'.join(nc_eval_slugs(nofeat=nofeat)))
"
)

echo "Submitting ${#DATASETS[@]} AnyGraph NC jobs (nofeat=${ANYGRAPH_NOFEAT})"
for dataset in "${DATASETS[@]}"; do
  job_name="ag-nc-${dataset}"
  echo "--- ${dataset} ---"
  ANYGRAPH_NC_DATASET="${dataset}" \
    ANYGRAPH_NOFEAT="${ANYGRAPH_NOFEAT}" \
    SLURM_JOB_NAME="${job_name}" \
    bash slurm_scripts/submit.sh "${JOB_SCRIPT}"
done
