#!/bin/bash
# Queue one GraphPFN PCA-64 finetune job per featured multiclass NC slug.
#
# Usage (from the repo root, login node):
#   bash slurm_scripts/baselines/submit_graphpfn_finetune.sh
#   GRAPHPFN_PCA_DIM=64 bash slurm_scripts/baselines/submit_graphpfn_finetune.sh

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
JOB_SCRIPT="${SLURM_SCRIPTS_DIR}/baselines/graphpfn_finetune.sh"
PCA_DIM="${GRAPHPFN_PCA_DIM:-64}"

mkdir -p logs

SUBMIT_PYTHON="$(require_env_python "${WANDER_ENV:-wander}" "${SUBMIT_PYTHON:-}")"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "=== Generating GraphPFN finetune configs (PCA-${PCA_DIM}, multiclass NC, no MUX) ==="
if [[ "${PCA_DIM}" == "none" || "${PCA_DIM}" == "nopca" ]]; then
  "${SUBMIT_PYTHON}" baselines/graphpfn/generate_finetune_configs.py --multiclass-nc --no-pca
else
  "${SUBMIT_PYTHON}" baselines/graphpfn/generate_finetune_configs.py --multiclass-nc --pca-dim "${PCA_DIM}"
fi

mapfile -t DATASETS < <(
  "${SUBMIT_PYTHON}" -c "
from baselines.registry import graphpfn_multiclass_nc_slugs
print('\n'.join(graphpfn_multiclass_nc_slugs(featured_only=True)))
"
)

echo "Submitting ${#DATASETS[@]} GraphPFN finetune jobs (pca=${PCA_DIM})"
for dataset in "${DATASETS[@]}"; do
  read -r SLURM_MEM < <(
    "${SUBMIT_PYTHON}" -c "
from baselines.registry import spec_by_slug
print(spec_by_slug('${dataset}').slurm_mem)
"
  )
  job_name="graphpfn-ft-${dataset}"
  echo "  ${dataset} (mem=${SLURM_MEM}) -> ${job_name}"
  GRAPHPFN_DATASET="${dataset}" \
    GRAPHPFN_PCA_DIM="${PCA_DIM}" \
    SLURM_JOB_NAME="${job_name}" \
    CLUSTER_MEM="${SLURM_MEM}" \
    bash slurm_scripts/submit.sh "${JOB_SCRIPT}"
done
