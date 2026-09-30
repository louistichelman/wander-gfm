#!/bin/bash
# Queue one GraphPFN ICL job per registry slug.
# Each job runs every Wander NC protocol split (official masks or GraphAny 0..4).
#
# Usage (from the repo root, login node):
#   bash slurm_scripts/baselines/submit_graphpfn_icl.sh
#   GRAPHPFN_PCA_DIM=none bash slurm_scripts/baselines/submit_graphpfn_icl.sh
#   GRAPHPFN_MULTICLASS_NC=1 bash slurm_scripts/baselines/submit_graphpfn_icl.sh  # paper NC set
#   GRAPHPFN_SPLIT_MODE=first bash slurm_scripts/baselines/submit_graphpfn_icl.sh  # split 0 only

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
JOB_SCRIPT="${SLURM_SCRIPTS_DIR}/baselines/graphpfn_icl.sh"
PCA_DIM="${GRAPHPFN_PCA_DIM:-64}"
NOFEAT="${GRAPHPFN_IGNORE_FEATURES:-0}"

mkdir -p logs

SUBMIT_PYTHON="$(require_env_python "${WANDER_ENV:-wander}" "${SUBMIT_PYTHON:-}")"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "=== Generating GraphPFN ICL configs (pca=${PCA_DIM}, nofeat=${NOFEAT}) ==="
if [[ "${NOFEAT}" == "1" || "${NOFEAT}" == "true" ]]; then
  "${SUBMIT_PYTHON}" baselines/graphpfn/generate_icl_configs.py --nofeat
elif [[ "${PCA_DIM}" == "none" || "${PCA_DIM}" == "nopca" || "${PCA_DIM}" == "0" ]]; then
  "${SUBMIT_PYTHON}" baselines/graphpfn/generate_icl_configs.py --no-pca
else
  "${SUBMIT_PYTHON}" baselines/graphpfn/generate_icl_configs.py --pca-dim "${PCA_DIM}"
fi

mapfile -t DATASETS < <(
  if [[ -n "${GRAPHPFN_SLUGS:-}" ]]; then
    echo "${GRAPHPFN_SLUGS}" | tr ' ,' '\n' | awk 'NF'
  elif [[ "${GRAPHPFN_MULTICLASS_NC:-0}" == "1" || "${GRAPHPFN_MULTICLASS_NC:-0}" == "true" ]]; then
    "${SUBMIT_PYTHON}" -c "
from baselines.registry import graphpfn_multiclass_nc_slugs
print('\n'.join(graphpfn_multiclass_nc_slugs(featured_only=True)))
"
  else
    "${SUBMIT_PYTHON}" -c "
from baselines.registry import graphpfn_slugs
print('\n'.join(graphpfn_slugs()))
"
  fi
)

echo "Submitting ${#DATASETS[@]} GraphPFN ICL jobs (all protocol splits)"
for dataset in "${DATASETS[@]}"; do
  job_name="graphpfn-icl-${dataset}"
  echo "--- ${dataset} ---"
  GRAPHPFN_DATASET="${dataset}" \
    SLURM_JOB_NAME="${job_name}" \
    bash slurm_scripts/submit.sh "${JOB_SCRIPT}"
done
