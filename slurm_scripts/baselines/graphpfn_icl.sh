#!/bin/bash
# GraphPFN ICL evaluation for one dataset slug (every Wander NC protocol split).
#
# Usage (from the repo root):
#   sbatch --export=ALL,GRAPHPFN_DATASET=cora slurm_scripts/baselines/graphpfn_icl.sh
#   GRAPHPFN_PCA_DIM=none sbatch --export=ALL,GRAPHPFN_DATASET=texas slurm_scripts/baselines/graphpfn_icl.sh
#   GRAPHPFN_SPLIT=3 sbatch --export=ALL,GRAPHPFN_DATASET=texas slurm_scripts/baselines/graphpfn_icl.sh

#SBATCH -J graphpfn-icl
#SBATCH --partition=gpu
#SBATCH --output=logs/graphpfn-icl-%j.out
#SBATCH --error=logs/graphpfn-icl-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

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

GRAPHPFN_ENV="${GRAPHPFN_ENV:-graphpfn_env}"
GRAPHPFN_PYTHON="$(require_env_python "${GRAPHPFN_ENV}" "${GRAPHPFN_PYTHON:-}")"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ "${GRAPHPFN_PREPARE:-0}" == "1" || "${GRAPHPFN_PREPARE:-}" == "true" ]]; then
  "${GRAPHPFN_PYTHON}" baselines/graphpfn/prepare.py --data_dir "${WANDER_DATA_DIR}" --slugs "${GRAPHPFN_DATASET}"
fi
"${GRAPHPFN_PYTHON}" baselines/graphpfn/run.py --dataset "${GRAPHPFN_DATASET}"
