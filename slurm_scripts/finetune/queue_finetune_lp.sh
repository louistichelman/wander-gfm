#!/usr/bin/env bash
# Login-node coordinator: queue ordinary (AnyGraph) LP finetunes from
# checkpoints/pretrained_wander/. Do NOT sbatch this file.
#
# Protocol (see finetune_lp.sh): 5 epochs, wd=0, 256 negatives,
# bs=4 featured / bs=8 no-feature, bpe=min(5000, ceil(n_train/bs)), skip_eval,
# PCA like zero-shot (featured 64/64, no-feature 64/32),
# featured 64x64 / no-feature 128x128.
# After FT, evaluate epoch 5: queue_eval_ft_lp.sh
#
# Usage (from the repo root):
#   bash slurm_scripts/finetune/queue_finetune_lp.sh
#   bash slurm_scripts/finetune/queue_finetune_lp.sh --dry-run
#   DATASETS="ANYGRAPH_CORA ANYGRAPH_DDI" bash slurm_scripts/finetune/queue_finetune_lp.sh

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: queue_finetune_lp.sh is a login-node coordinator — run with bash, not sbatch." >&2
  exit 1
fi

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" || "${1:-}" == "-n" ]]; then
  DRY_RUN=1
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
mkdir -p logs checkpoints raw_data
# shellcheck disable=SC2206
DATASETS=(${DATASETS:-${PAPER_LP_DATASETS[*]}})
FT_GPUS="${FT_GPUS:-1}"

dataset_mem() {
  local ds="$1"
  case "${ds}" in
    ANYGRAPH_DDI|ANYGRAPH_PRODUCTS_HOME|ANYGRAPH_PROTEINS_SPEC1|ANYGRAPH_SOC_EPINIONS1|ANYGRAPH_EMAIL_ENRON|ANYGRAPH_PUBMED|ANYGRAPH_CS)
      echo "${LARGE_LP_MEM:-128G}" ;;
    *) echo "${CLUSTER_MEM:-64G}" ;;
  esac
}

submit_job() {
  local ds="$1"
  local mem
  mem="$(dataset_mem "${ds}")"
  local -a cmd=(
    env
    "SLURM_NUM_GPUS=${FT_GPUS}"
    "SLURM_JOB_NAME=ft-lp-${ds}"
    "CLUSTER_MEM=${mem}"
    "CLUSTER_TIME_FINETUNE=${CLUSTER_TIME_FINETUNE:-2-00:00:00}"
    "RUN_NAME=${RUN_NAME_PREFIX:-ft}_${ds}_lp"
    "MAX_EPOCHS=${MAX_EPOCHS:-5}"
    "FORCE_FRESH=${FORCE_FRESH:-0}"
    bash "${SCRIPT_DIR}/submit.sh" "${SCRIPT_DIR}/finetune/finetune_lp.sh" "${ds}"
  )
  printf '+ '
  printf '%q ' "${cmd[@]}"
  echo
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${cmd[@]}"
  fi
}

echo "=== Ordinary LP finetune (5 ep, bpe=min(5000, n_train/bs), bs=4 feat / 8 nofeat, skip_eval, ZS PCA, wd=0, finetune) ==="
echo "Repo: ${REPO_ROOT}"
echo "Datasets: ${DATASETS[*]}"
echo "After FT (epoch 5): bash slurm_scripts/eval/queue_eval_ft_lp.sh"

for ds in "${DATASETS[@]}"; do
  echo "--- ft_${ds}_lp mem=$(dataset_mem "${ds}") ---"
  submit_job "${ds}"
done

echo "=== Done queueing (dry_run=${DRY_RUN}) ==="
echo "Monitor: squeue -u \$USER"
