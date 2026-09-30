#!/usr/bin/env bash
#SBATCH --job-name=gcn-grid
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm_gcn_grid_%x_%j.out
#SBATCH --error=logs/slurm_gcn_grid_%x_%j.err

# CPU GCN hyperparameter grid search for node-classification datasets.
# Default: 60-config grid on split 0, then that best config on later Wander
# protocol splits. Summarize averages the per-split scores.
#
# Usage (from the repo root):
#   sbatch slurm_scripts/baselines/train_gcn_grid.sh MOTIF_MIXED
#   sbatch --job-name=gcn-MOTIF_MIXED slurm_scripts/baselines/train_gcn_grid.sh MOTIF_MIXED
#   GCN_GRID_SPLIT=0 sbatch slurm_scripts/baselines/train_gcn_grid.sh TEXAS
#
# Env overrides:
#   GCN_GRID_OUTPUT_DIR  default: ${RESULTS_DIR}/gcn_grid (or gcn_grid_nofeat)
#   GCN_GRID_HIDDEN_DIM  default: 128
#   GCN_GRID_MAX_EPOCHS  default: 1000
#   GCN_GRID_PATIENCE    default: 50
#   GCN_GRID_IGNORE_FEATURES  default: 0 (set to 1 for structure-only / nofeat grids)
#   GCN_GRID_SPLIT       optional single protocol split index
#   GCN_GRID_SPLIT_MODE  all (default) or first
#   GCN_GRID_HPARAM_MODE first (default) or per-split
#   GCN_GRID_OLD_BEST    set to 1 to skip the grid and use old_best_settings.md
#   GCN_GRID_DEVICE      default: cpu (set to cuda to train on the allocated GPU)

set -euo pipefail
export PYTHONUNBUFFERED=1

DATASET="${1:-${GCN_GRID_DATASET:-}}"
if [[ -z "${DATASET}" ]]; then
  echo "ERROR: pass dataset registry key as first argument or set GCN_GRID_DATASET" >&2
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
source "${_slurm}/source_common.sh"

OUTPUT_DIR="${GCN_GRID_OUTPUT_DIR:-${RESULTS_DIR}/gcn_grid}"
HIDDEN_DIM="${GCN_GRID_HIDDEN_DIM:-128}"
MAX_EPOCHS="${GCN_GRID_MAX_EPOCHS:-1000}"
PATIENCE="${GCN_GRID_PATIENCE:-50}"
IGNORE_FEATURES="${GCN_GRID_IGNORE_FEATURES:-0}"
SPLIT_MODE="${GCN_GRID_SPLIT_MODE:-all}"
HPARAM_MODE="${GCN_GRID_HPARAM_MODE:-first}"
DEVICE="${GCN_GRID_DEVICE:-cpu}"

if [[ "${IGNORE_FEATURES}" == "1" ]]; then
  if [[ "${OUTPUT_DIR}" == "${RESULTS_DIR}/gcn_grid" ]]; then
    OUTPUT_DIR="${RESULTS_DIR}/gcn_grid_nofeat"
  fi
fi

IGNORE_ARGS=()
if [[ "${IGNORE_FEATURES}" == "1" ]]; then
  IGNORE_ARGS+=(--ignore_features)
fi

SPLIT_ARGS=(--split-mode "${SPLIT_MODE}" --hparam-mode "${HPARAM_MODE}")
if [[ -n "${GCN_GRID_SPLIT:-}" ]]; then
  SPLIT_ARGS+=(--split "${GCN_GRID_SPLIT}")
fi
if [[ "${GCN_GRID_OLD_BEST:-0}" == "1" ]]; then
  SPLIT_ARGS+=(--old-best)
fi

mkdir -p "${OUTPUT_DIR}" "${LOGS_DIR}"
cluster_stage_datasets "${DATASET}"

echo "=== GCN grid search (${DEVICE}): ${DATASET} ==="
echo "Data dir: ${RAW_DATA_DIR}"
echo "Output:   ${OUTPUT_DIR}/${DATASET,,}$([[ "${IGNORE_FEATURES}" == "1" ]] && echo _nofeat)"
echo "Grid:     60 configs on split 0; later splits reuse that best config"
echo "Splits:   mode=${SPLIT_MODE} split=${GCN_GRID_SPLIT:-all} hparam_mode=${HPARAM_MODE} old_best=${GCN_GRID_OLD_BEST:-0}"
echo "Device:   ${DEVICE}"
echo "No-feat:  ${IGNORE_FEATURES}"

"${PYTHON}" "${REPO_ROOT}/scripts/baselines/train_gcn_grid.py" \
  --datasets "${DATASET}" \
  --data_dir "${RAW_DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --seed 42 \
  --hidden_dim "${HIDDEN_DIM}" \
  --num_layers 3 \
  --max_epochs "${MAX_EPOCHS}" \
  --patience "${PATIENCE}" \
  --weight_decay 5e-4 \
  --device "${DEVICE}" \
  --resume \
  "${SPLIT_ARGS[@]}" \
  "${IGNORE_ARGS[@]}"

finish_job
