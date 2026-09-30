#!/bin/bash
# AnyGraph node-classification eval for one Wander slug (pretrain_link2).
#
# Usage (from the repo root):
#   ANYGRAPH_NC_DATASET=cornell bash slurm_scripts/submit.sh \
#     slurm_scripts/baselines/anygraph_nc.sh
#   ANYGRAPH_NOFEAT=1 ANYGRAPH_NC_DATASET=cornell bash slurm_scripts/submit.sh \
#     slurm_scripts/baselines/anygraph_nc.sh

#SBATCH -J anygraph-nc
#SBATCH --partition=gpu
#SBATCH --output=logs/anygraph-nc-%j.out
#SBATCH --error=logs/anygraph-nc-%j.err
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

if [[ -z "${ANYGRAPH_NC_DATASET:-}" ]]; then
  echo "ERROR: ANYGRAPH_NC_DATASET must be set" >&2
  exit 1
fi

ANYGRAPH_ENV="${ANYGRAPH_MAMBA_ENV:-anygraph_env}"
WANDER_PYTHON="$(require_env_python "${WANDER_ENV}" "${WANDER_PYTHON:-}")"
ANYGRAPH_PYTHON="$(require_env_python "${ANYGRAPH_ENV}" "${ANYGRAPH_PYTHON:-}")"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -z "${ANYGRAPH_NC_DATA_ROOT:-}" ]]; then
  export ANYGRAPH_NC_DATA_ROOT="${REPO_ROOT}/raw_data/anygraph_nc"
fi
NOFEAT_FLAG=()
if [[ "${ANYGRAPH_NOFEAT:-0}" == "1" ]]; then
  NOFEAT_FLAG=(--nofeat)
fi

"${WANDER_PYTHON}" baselines/anygraph/prepare_nc.py \
  --data_dir "${WANDER_DATA_DIR}" \
  --dataset "${ANYGRAPH_NC_DATASET}" \
  "${NOFEAT_FLAG[@]}"

"${ANYGRAPH_PYTHON}" -c "import setproctitle" 2>/dev/null || "${ANYGRAPH_PYTHON}" -m pip install -q setproctitle
"${ANYGRAPH_PYTHON}" baselines/anygraph/run.py \
  --load_model "${ANYGRAPH_NC_CKPT:-pretrain_link2}" \
  --dataset "${ANYGRAPH_NC_DATASET}" \
  --eval_type node \
  "${NOFEAT_FLAG[@]}"
