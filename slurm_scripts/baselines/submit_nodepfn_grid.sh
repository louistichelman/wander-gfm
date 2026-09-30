#!/bin/bash
# Queue NodePFN TSVD {10,15,20,25,30} × smoothing {0,1} on paper NC graphs
# (no official run.sh mapping). Reports: results/baselines/nodepfn/nc-grid/
#
# Usage (from the repo root, login node):
#   bash slurm_scripts/baselines/submit_nodepfn_grid.sh
#   DATASETS="roman-empire motif-mixed" bash slurm_scripts/baselines/submit_nodepfn_grid.sh

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
JOB_SCRIPT="${SLURM_SCRIPTS_DIR}/baselines/nodepfn_eval.sh"

mkdir -p logs

if [[ -n "${DATASETS:-}" ]]; then
  # shellcheck disable=SC2206
  DATASET_LIST=(${DATASETS})
else
  SUBMIT_PYTHON="$(require_env_python "${WANDER_ENV:-wander}" "${SUBMIT_PYTHON:-}")"
  mapfile -t DATASET_LIST < <(
    PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${SUBMIT_PYTHON}" -c "
from baselines.nodepfn.configs import featured_slugs_for_tsvd_grid
print('\n'.join(featured_slugs_for_tsvd_grid()))
"
  )
fi

echo "Submitting ${#DATASET_LIST[@]} NodePFN TSVD/smoothing grid jobs"
for dataset in "${DATASET_LIST[@]}"; do
  mem="${CLUSTER_MEM:-128G}"
  batch="${NODEPFN_BATCH_SIZE_INFERENCE:-32}"
  time_limit="${CLUSTER_TIME_EVAL:-0-24:00:00}"
  case "${dataset}" in
    pokec-regions-100k-top10)
      mem="${CLUSTER_MEM:-256G}"
      batch="${NODEPFN_BATCH_SIZE_INFERENCE:-1}"
      ;;
    roman-empire)
      batch="${NODEPFN_BATCH_SIZE_INFERENCE:-1}"
      ;;
  esac
  job_name="nodepfn-grid-${dataset}"
  echo "--- ${dataset} mem=${mem} batch=${batch} ---"
  NODEPFN_DATASET="${dataset}" \
    NODEPFN_GRID=1 \
    NODEPFN_BATCH_SIZE_INFERENCE="${batch}" \
    SLURM_JOB_NAME="${job_name}" \
    CLUSTER_MEM="${mem}" \
    CLUSTER_TIME_EVAL="${time_limit}" \
    bash slurm_scripts/submit.sh "${JOB_SCRIPT}"
done
