#!/bin/bash
# NodePFN inference on one NC dataset (raw tensors, Wander splits, NodePFN preprocess).
#
# Usage (from the repo root):
#   sbatch --export=ALL,NODEPFN_DATASET=cora slurm_scripts/baselines/nodepfn_eval.sh
#   bash slurm_scripts/submit.sh slurm_scripts/baselines/nodepfn_eval.sh
#     (export NODEPFN_DATASET first)
#
# Prepare first (wander env):
#   python baselines/nodepfn/prepare.py --dataset cora

#SBATCH -J nodepfn-eval
#SBATCH --partition=gpu
#SBATCH --output=logs/nodepfn-eval-%j.out
#SBATCH --error=logs/nodepfn-eval-%j.err
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

if [[ -z "${NODEPFN_DATASET:-}" ]]; then
  echo "ERROR: NODEPFN_DATASET must be set" >&2
  exit 1
fi

NODEPFN_ENV="${NODEPFN_ENV:-nodepfn_env}"
export PATH="${HOME}/.local/bin:${HOME}/micromamba/bin:${PATH}"
NODEPFN_PYTHON="$(require_env_python "${NODEPFN_ENV}" "${NODEPFN_PYTHON:-}")"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cmd=("${NODEPFN_PYTHON}" baselines/nodepfn/run.py --dataset "${NODEPFN_DATASET}")
if [[ "${NODEPFN_NOFEAT:-0}" == "1" ]]; then
  cmd+=(--nofeat)
fi
if [[ "${NODEPFN_CPU:-0}" == "1" ]]; then
  cmd+=(--cpu)
fi
if [[ -n "${NODEPFN_RUNS:-}" ]]; then
  cmd+=(--runs "${NODEPFN_RUNS}")
fi
if [[ -n "${NODEPFN_SPLIT:-}" ]]; then
  cmd+=(--split "${NODEPFN_SPLIT}")
fi
if [[ -n "${NODEPFN_SPLIT_MODE:-}" ]]; then
  cmd+=(--split-mode "${NODEPFN_SPLIT_MODE}")
fi
if [[ -n "${NODEPFN_BATCH_SIZE_INFERENCE:-}" ]]; then
  cmd+=(--batch-size-inference "${NODEPFN_BATCH_SIZE_INFERENCE}")
fi
if [[ "${NODEPFN_GRID:-0}" == "1" ]]; then
  cmd+=(--grid)
fi
if [[ -n "${NODEPFN_N_COMPONENTS:-}" ]]; then
  cmd+=(--n-components "${NODEPFN_N_COMPONENTS}")
fi
if [[ -n "${NODEPFN_SMOOTHING_STEPS:-}" ]]; then
  cmd+=(--smoothing-steps "${NODEPFN_SMOOTHING_STEPS}")
fi
printf '+ '
printf '%q ' "${cmd[@]}"
echo
"${cmd[@]}"
