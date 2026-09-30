#!/bin/bash
# AnyGraph zero-shot eval (pretrained checkpoint, epoch 0).
#
# Usage (from the repo root):
#   sbatch slurm_scripts/baselines/anygraph_eval.sh pretrain_link1 link2 link
#   EVAL_PROTOCOL_WANDER=1 sbatch slurm_scripts/baselines/anygraph_eval.sh pretrain_link1 link2 link

#SBATCH -J anygraph-eval
#SBATCH --partition=gpu
#SBATCH --output=logs/anygraph-eval-%j.out
#SBATCH --error=logs/anygraph-eval-%j.err
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

LOAD_MODEL="${1:?Usage: sbatch anygraph_eval.sh <load_model> <dataset_setting> [link|node]}"
DATASET_SETTING="${2:?Missing dataset_setting}"
EVAL_TYPE="${3:-link}"

ANYGRAPH_ENV="${ANYGRAPH_MAMBA_ENV:-anygraph_env}"
ANYGRAPH_PYTHON="$(require_env_python "${ANYGRAPH_ENV}" "${ANYGRAPH_PYTHON:-}")"

"${ANYGRAPH_PYTHON}" -c "import setproctitle" 2>/dev/null || "${ANYGRAPH_PYTHON}" -m pip install -q setproctitle

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -z "${ANYGRAPH_DATA_ROOT:-}" && -d "${REPO_ROOT}/raw_data/AnyGraph/zero-shot datasets" ]]; then
  export ANYGRAPH_DATA_ROOT="${REPO_ROOT}/raw_data/AnyGraph/zero-shot datasets"
fi
WANDER_FLAG=()
if [[ "${EVAL_PROTOCOL_WANDER:-0}" == "1" ]]; then
  WANDER_FLAG=(--eval_protocol_wander)
fi

"${ANYGRAPH_PYTHON}" baselines/anygraph/run.py \
  --load_model "${LOAD_MODEL}" \
  --dataset_setting "${DATASET_SETTING}" \
  --eval_type "${EVAL_TYPE}" \
  "${WANDER_FLAG[@]}"
