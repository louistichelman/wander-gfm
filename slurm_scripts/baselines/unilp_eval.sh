#!/bin/bash
# Official UniLP (context_LP) zero-shot inference.
#
# Usage (from the repo root):
#   sbatch slurm_scripts/baselines/unilp_eval.sh
#   UNILP_DATASETS=USAir,NS sbatch slurm_scripts/baselines/unilp_eval.sh

#SBATCH -J unilp-eval
#SBATCH --partition=gpu
#SBATCH --output=logs/unilp-eval-%j.out
#SBATCH --error=logs/unilp-eval-%j.err
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

UNILP_ENV="${UNILP_ENV:-unilp_env}"
UNILP_PYTHON="$(require_env_python "${UNILP_ENV}" "${UNILP_PYTHON:-}")"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -z "${UNILP_DATA_DIR:-}" && -d "${REPO_ROOT}/third_party/context_LP/data" ]]; then
  export UNILP_DATA_DIR="${REPO_ROOT}/third_party/context_LP/data"
fi

CMD=("${UNILP_PYTHON}" baselines/unilp/run.py)
if [[ -n "${UNILP_DATASETS:-}" ]]; then
  CMD+=(--datasets "${UNILP_DATASETS}")
fi
if [[ -n "${UNILP_CKPT:-}" ]]; then
  CMD+=(--load_model "${UNILP_CKPT}")
fi
if [[ -n "${UNILP_K:-}" ]]; then
  CMD+=(--k "${UNILP_K}")
fi
"${CMD[@]}"
