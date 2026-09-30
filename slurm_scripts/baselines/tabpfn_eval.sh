#!/bin/bash
# TabPFNv2 inference on one NC dataset (tabular features, PCA 64, Wander splits).
#
# Usage (from the repo root):
#   sbatch --export=ALL,TABPFN_DATASET=cora slurm_scripts/baselines/tabpfn_eval.sh
#   bash slurm_scripts/submit.sh slurm_scripts/baselines/tabpfn_eval.sh
#
# Prepare first (wander env), or reuse nodepfn_nc via TABPFN_DATA_ROOT:
#   python baselines/tabpfn/prepare.py --dataset cora

#SBATCH -J tabpfn-eval
#SBATCH --partition=gpu
#SBATCH --output=logs/tabpfn-eval-%j.out
#SBATCH --error=logs/tabpfn-eval-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

set -euo pipefail

_common="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common.sh"
if [[ ! -f "${_common}" ]]; then
  _common="${SLURM_SUBMIT_DIR:-}/slurm_scripts/common.sh"
fi
# shellcheck source=/dev/null
source "${_common}"

# Load TABPFN_TOKEN / HF secrets from repo .env when present.
_env_file="${REPO_ROOT}/.env"
if [[ -f "${_env_file}" ]]; then
  # shellcheck source=/dev/null
  set -a
  # shellcheck disable=SC1090
  source "${_env_file}"
  set +a
  if [[ -n "${TABPFN_TOKEN:-}" ]]; then
    echo "Loaded TABPFN_TOKEN from ${_env_file}"
  fi
fi

if [[ -z "${TABPFN_DATASET:-}" ]]; then
  echo "ERROR: TABPFN_DATASET must be set" >&2
  exit 1
fi

TABPFN_ENV="${TABPFN_ENV:-tabpfn_env}"
export PATH="${HOME}/.local/bin:${HOME}/micromamba/bin:${PATH}"
TABPFN_PYTHON="$(require_env_python "${TABPFN_ENV}" "${TABPFN_PYTHON:-}")"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cmd=("${TABPFN_PYTHON}" baselines/tabpfn/run.py --dataset "${TABPFN_DATASET}")
if [[ "${TABPFN_NOFEAT:-0}" == "1" ]]; then
  cmd+=(--nofeat)
fi
if [[ "${TABPFN_CPU:-0}" == "1" ]]; then
  cmd+=(--cpu)
fi
if [[ -n "${TABPFN_PCA_DIM:-}" ]]; then
  cmd+=(--pca-dim "${TABPFN_PCA_DIM}")
fi
if [[ -n "${TABPFN_MODEL_VERSION:-}" ]]; then
  cmd+=(--model-version "${TABPFN_MODEL_VERSION}")
fi
if [[ -n "${TABPFN_SEED:-}" ]]; then
  cmd+=(--seed "${TABPFN_SEED}")
fi
if [[ -n "${TABPFN_N_ESTIMATORS:-}" ]]; then
  cmd+=(--n-estimators "${TABPFN_N_ESTIMATORS}")
fi
if [[ -n "${TABPFN_SPLIT:-}" ]]; then
  cmd+=(--split "${TABPFN_SPLIT}")
fi
if [[ -n "${TABPFN_SPLIT_MODE:-}" ]]; then
  cmd+=(--split-mode "${TABPFN_SPLIT_MODE}")
fi
if [[ "${TABPFN_RESPECT_LIMITS:-0}" == "1" ]]; then
  cmd+=(--respect-pretraining-limits)
fi
printf '+ '
printf '%q ' "${cmd[@]}"
echo
"${cmd[@]}"
