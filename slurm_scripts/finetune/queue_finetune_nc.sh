#!/usr/bin/env bash
# Login-node coordinator: queue NC finetune LR-grid jobs from
# checkpoints/pretrained_wander/. Do NOT sbatch this file.
#
# Usage (from the repo root):
#   bash slurm_scripts/finetune/queue_finetune_nc.sh
#   bash slurm_scripts/finetune/queue_finetune_nc.sh --dry-run
#   DATASETS="CORA SQUIRREL_FILTERED" bash slurm_scripts/finetune/queue_finetune_nc.sh
#   LR=1.08e-4 DATASETS="CORA" bash slurm_scripts/finetune/queue_finetune_nc.sh
#
# Protocol (see finetune_nc.sh): first split only, skip_final_test,
# bs=16, bpe=5, query 0.25/1024, epochs=100, patience=15, wd=0, val cap=2000,
# official 10-point LR grid. Val: wander_test_samples=3 + train_kv_passes=3.
# After the grid: bash slurm_scripts/eval/queue_eval_ft_nc.sh

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: queue_finetune_nc.sh is a login-node coordinator — run with bash, not sbatch." >&2
  exit 1
fi

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
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
# Official GraphPFN-style 10-point LR grid.
NC_LR_GRID=(
  5.00e-6 8.34e-6 1.39e-5 2.32e-5 3.87e-5
  6.46e-5 1.08e-4 1.80e-4 3.00e-4 5.00e-4
)
# shellcheck disable=SC2206
DATASETS=(${DATASETS:-${PAPER_NC_DATASETS[*]}})
if [[ -n "${LR:-}" && -z "${LR_GRID:-}" ]]; then
  LRS=("${LR}")
else
  # shellcheck disable=SC2206
  LRS=(${LR_GRID:-${NC_LR_GRID[*]}})
fi
FT_GPUS="${FT_GPUS:-1}"

lr_tag() {
  echo "$1" | tr '.' 'p'
}

dataset_mem() {
  local ds="$1"
  case "${ds}" in
    CO_PHYSICS|HM_CATEGORIES|FULL_CORA|FULL_DBLP|POKEC_REGIONS_100K|POKEC_REGIONS_100K_TOP10|CITY_PARIS|CITY_SHANGHAI|CITY_LA)
      echo "${LARGE_NC_MEM:-256G}" ;;
    AMAZON_RATINGS|COMPUTERS|CO_CS|PUBMED|ROMAN_EMPIRE|WIKI_CS)
      echo "${MID_NC_MEM:-128G}"
      ;;
    *) echo "${CLUSTER_MEM:-64G}" ;;
  esac
}

submit_job() {
  local ds="$1"
  local lr="$2"
  local mem tag run_name
  mem="$(dataset_mem "${ds}")"
  tag="$(lr_tag "${lr}")"
  run_name="${RUN_NAME_PREFIX:-ft}_${ds}_nc_lr${tag}"
  local -a cmd=(
    env
    "SLURM_NUM_GPUS=${FT_GPUS}"
    "SLURM_JOB_NAME=ft-nc-${ds}-${tag}"
    "CLUSTER_MEM=${mem}"
    "CLUSTER_TIME_FINETUNE=${CLUSTER_TIME_FINETUNE:-2-00:00:00}"
    "RUN_NAME=${run_name}"
    "LR=${lr}"
    "PATIENCE=${PATIENCE:-15}"
    "BATCH_PER_EPOCH=${BATCH_PER_EPOCH:-5}"
    "WANDER_TRAIN_KV_PASSES=${WANDER_TRAIN_KV_PASSES:-3}"
    "FINAL_TEST_TRAIN_KV_PASSES=${FINAL_TEST_TRAIN_KV_PASSES:-16}"
    "WANDER_TRAIN_KV_WALK_NUM=${WANDER_TRAIN_KV_WALK_NUM:-}"
    "NC_SPLIT_INDEX=${NC_SPLIT_INDEX:-0}"
    "SKIP_FINAL_TEST=${SKIP_FINAL_TEST:-1}"
    "FORCE_FRESH=${FORCE_FRESH:-0}"
    bash "${SCRIPT_DIR}/submit.sh" "${SCRIPT_DIR}/finetune/finetune_nc.sh" "${ds}"
  )
  printf '+ '
  printf '%q ' "${cmd[@]}"
  echo
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${cmd[@]}"
  fi
}

n_jobs=$(( ${#DATASETS[@]} * ${#LRS[@]} ))
echo "=== NC finetune LR grid (features on, split 0, skip_final_test, ${FT_GPUS} GPU, ${n_jobs} jobs) ==="
echo "Repo: ${REPO_ROOT}"
echo "Datasets: ${DATASETS[*]}"
echo "LRs: ${LRS[*]}"
echo "Protocol: split=0 bs=16 bpe=5 query=0.25/1024 epochs=100 patience=${PATIENCE:-15} wd=0 val_cap=2000 walks 128 x Table-8 adaptive length kv/ts 3 val; no final test"
echo "After grid: bash slurm_scripts/eval/queue_eval_ft_nc.sh"

for ds in "${DATASETS[@]}"; do
  for lr in "${LRS[@]}"; do
    echo "--- ${ds} lr=${lr} mem=$(dataset_mem "${ds}") ---"
    submit_job "${ds}" "${lr}"
  done
done

echo "=== Done queueing (dry_run=${DRY_RUN}) ==="
echo "Monitor: squeue -u \$USER"
