#!/bin/bash
# GraphPFN LR-grid finetune for one dataset slug (PCA-64 by default).
#
# Usage (from the repo root):
#   sbatch --export=ALL,GRAPHPFN_DATASET=cora slurm_scripts/baselines/graphpfn_finetune.sh

#SBATCH -J graphpfn-ft
#SBATCH --partition=gpu
#SBATCH --output=logs/graphpfn-ft-%j.out
#SBATCH --error=logs/graphpfn-ft-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

set -euo pipefail

_common="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
if [[ ! -f "${_common}" ]]; then
  _common="${SLURM_SUBMIT_DIR:-}/slurm_scripts/baselines/common.sh"
fi
# shellcheck source=/dev/null
source "${_common}"

if [[ -z "${GRAPHPFN_DATASET:-}" ]]; then
  echo "ERROR: GRAPHPFN_DATASET must be set" >&2
  exit 1
fi

GRAPHPFN_PCA_DIM="${GRAPHPFN_PCA_DIM:-64}"
GRAPHPFN_ENV="${GRAPHPFN_ENV:-graphpfn_env}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
WANDER_PYTHON="$(require_env_python "${WANDER_ENV}" "${WANDER_PYTHON:-}")"
GRAPHPFN_PYTHON="$(require_env_python "${GRAPHPFN_ENV}" "${GRAPHPFN_PYTHON:-}")"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${SKIP_PREPARE}" != "1" ]]; then
  echo "=== Preparing GraphPFN data for ${GRAPHPFN_DATASET} ==="
  "${WANDER_PYTHON}" baselines/graphpfn/prepare.py \
    --data_dir "${WANDER_DATA_DIR}" \
    --slugs "${GRAPHPFN_DATASET}"
fi

echo "=== GraphPFN finetune ${GRAPHPFN_DATASET} pca=${GRAPHPFN_PCA_DIM} ==="
"${GRAPHPFN_PYTHON}" baselines/graphpfn/run.py --dataset "${GRAPHPFN_DATASET}" --finetune --pca-dim "${GRAPHPFN_PCA_DIM}"
