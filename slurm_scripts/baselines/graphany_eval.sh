#!/bin/bash
# GraphAny inference on one NC dataset (raw tensors, Wander splits).
#
# Usage (from the repo root):
#   sbatch --export=ALL,GRAPHANY_DATASET=cora slurm_scripts/baselines/graphany_eval.sh
#   bash slurm_scripts/submit.sh slurm_scripts/baselines/graphany_eval.sh
#     (export GRAPHANY_DATASET first)
#
# Prepare first (wander env):
#   python baselines/graphany/prepare.py --dataset cora

#SBATCH -J graphany-eval
#SBATCH --partition=gpu
#SBATCH --output=logs/graphany-eval-%j.out
#SBATCH --error=logs/graphany-eval-%j.err
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

if [[ -z "${GRAPHANY_DATASET:-}" ]]; then
  echo "ERROR: GRAPHANY_DATASET must be set" >&2
  exit 1
fi

GRAPHANY_ENV="${GRAPHANY_ENV:-graphany_env}"
export PATH="${HOME}/.local/bin:${HOME}/micromamba/bin:${PATH}"
GRAPHANY_PYTHON="$(require_env_python "${GRAPHANY_ENV}" "${GRAPHANY_PYTHON:-}")"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export DGLBACKEND="${DGLBACKEND:-pytorch}"
export WANDB_MODE="${WANDB_MODE:-offline}"
cmd=("${GRAPHANY_PYTHON}" baselines/graphany/run.py --dataset "${GRAPHANY_DATASET}")
if [[ "${GRAPHANY_NOFEAT:-0}" == "1" ]]; then
  cmd+=(--nofeat)
fi
if [[ "${GRAPHANY_CPU:-0}" == "1" ]]; then
  cmd+=(--cpu)
fi
if [[ -n "${GRAPHANY_CKPT:-}" ]]; then
  cmd+=(--ckpt "${GRAPHANY_CKPT}")
fi
if [[ -n "${GRAPHANY_SPLIT:-}" ]]; then
  cmd+=(--split "${GRAPHANY_SPLIT}")
fi
if [[ -n "${GRAPHANY_SPLIT_MODE:-}" ]]; then
  cmd+=(--split-mode "${GRAPHANY_SPLIT_MODE}")
fi
printf '+ '
printf '%q ' "${cmd[@]}"
echo
"${cmd[@]}"
