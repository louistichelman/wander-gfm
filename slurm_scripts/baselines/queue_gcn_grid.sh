#!/usr/bin/env bash
# Login-node coordinator: queue one GCN grid job per NC dataset.
# Default: 60-config grid on split 0, then that best config on later splits.
#
# Usage (from the repo root):
#   bash slurm_scripts/baselines/queue_gcn_grid.sh
#   DATASETS="CORA TEXAS" bash slurm_scripts/baselines/queue_gcn_grid.sh
#   GCN_GRID_IGNORE_FEATURES=1 bash slurm_scripts/baselines/queue_gcn_grid.sh
#   GCN_GRID_ONE_JOB_PER_SPLIT=1 bash slurm_scripts/baselines/queue_gcn_grid.sh
#
# After jobs finish:
#   python scripts/baselines/summarize_gcn_grid.py

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
# shellcheck source=/dev/null
source "${SLURM_SCRIPTS_DIR}/paper_protocol.sh"
SCRIPT_DIR="${SLURM_SCRIPTS_DIR}"
cd "${REPO_ROOT}"
JOB_SCRIPT="${SCRIPT_DIR}/baselines/train_gcn_grid.sh"
NOFEAT="${GCN_GRID_IGNORE_FEATURES:-0}"
ONE_PER_SPLIT="${GCN_GRID_ONE_JOB_PER_SPLIT:-0}"

mkdir -p logs

# shellcheck disable=SC2206
DATASET_LIST=(${DATASETS:-${PAPER_NC_DATASETS[*]}})

echo "Submitting ${#DATASET_LIST[@]} GCN grid dataset(s) (nofeat=${NOFEAT}, one_job_per_split=${ONE_PER_SPLIT})"
for dataset in "${DATASET_LIST[@]}"; do
  if [[ "${ONE_PER_SPLIT}" == "1" ]]; then
    mapfile -t SPLITS < <(python data/nc_splits.py "${dataset}")
    for split in "${SPLITS[@]}"; do
      job_name="gcn-${dataset}-s${split}"
      echo "--- ${dataset} split ${split} ---"
      GCN_GRID_DATASET="${dataset}" \
        GCN_GRID_IGNORE_FEATURES="${NOFEAT}" \
        GCN_GRID_SPLIT="${split}" \
        SLURM_JOB_NAME="${job_name}" \
        bash slurm_scripts/submit.sh "${JOB_SCRIPT}" "${dataset}"
    done
  else
    job_name="gcn-${dataset}"
    echo "--- ${dataset} ---"
    GCN_GRID_DATASET="${dataset}" \
      GCN_GRID_IGNORE_FEATURES="${NOFEAT}" \
      SLURM_JOB_NAME="${job_name}" \
      bash slurm_scripts/submit.sh "${JOB_SCRIPT}" "${dataset}"
  fi
done
