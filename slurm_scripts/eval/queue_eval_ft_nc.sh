#!/usr/bin/env bash
# Login-node coordinator: zero-shot test of the best NC finetune checkpoint
# per dataset (highest val score across the LR grid).
# Same walk budget and seeds as queue_eval_nc.sh, but split 0 only.
# Do NOT sbatch this file.
#
# Usage (from the repo root):
#   bash slurm_scripts/eval/queue_eval_ft_nc.sh
#   bash slurm_scripts/eval/queue_eval_ft_nc.sh --dry-run
#   DATASETS="CORA SQUIRREL_FILTERED" bash slurm_scripts/eval/queue_eval_ft_nc.sh

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: queue_eval_ft_nc.sh is a login-node coordinator — run with bash, not sbatch." >&2
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
mkdir -p logs checkpoints raw_data results/eval

# shellcheck disable=SC2206
DATASETS=(${DATASETS:-${PAPER_NC_DATASETS[*]}})
NC_SPLIT_INDEX="${NC_SPLIT_INDEX:-0}"
EVAL_SEEDS="${EVAL_SEEDS:-0:1:2}"
SEED="${SEED:-0}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-ft}"

EVAL_RUN_DIR="${EVAL_RUN_DIR:-${REPO_ROOT}/results/eval/ft_nc_$(date +%y%m%d_%H%M)}"
JOB_SCRIPT="${JOB_SCRIPT:-${SCRIPT_DIR}/eval/eval_one_dataset.sh}"
PICK_SCRIPT="${SCRIPT_DIR}/finetune/pick_best_nc_ckpt.py"

dataset_mem() {
  local ds="$1"
  case "${ds}" in
    CO_PHYSICS|HM_CATEGORIES|FULL_CORA|FULL_DBLP|POKEC_REGIONS_100K|POKEC_REGIONS_100K_TOP10|CITY_PARIS|CITY_SHANGHAI|CITY_LA)
      echo "${LARGE_NC_MEM:-256G}" ;;
    AMAZON_RATINGS|COMPUTERS|CO_CS|PUBMED|ROMAN_EMPIRE|WIKI_CS)
      echo "${MID_NC_MEM:-128G}" ;;
    *) echo "${CLUSTER_MEM:-64G}" ;;
  esac
}


dataset_time() {
  local ds="$1"
  case "${ds}" in
    CO_PHYSICS|HM_CATEGORIES|FULL_CORA|FULL_DBLP|POKEC_REGIONS_100K|POKEC_REGIONS_100K_TOP10|CITY_PARIS|CITY_SHANGHAI|CITY_LA)
      echo "${CLUSTER_TIME_EVAL:-2-00:00:00}" ;;
    *) echo "${CLUSTER_TIME_EVAL:-1-00:00:00}" ;;
  esac
}

pick_ckpt() {
  local ds="$1"
  python3 "${PICK_SCRIPT}" "${ds}" \
    --checkpoints-dir "${REPO_ROOT}/checkpoints" \
    --run-prefix "${RUN_NAME_PREFIX}" \
    --seed "${SEED}"
}

submit_job() {
  local ds="$1"
  local ckpt="$2"
  local score="$3"
  local grid_run="$4"
  local mem
  mem="$(dataset_mem "${ds}")"
  local ignore_features="${IGNORE_FEATURES:-0}"
  local -a cmd=(
    env
    "SLURM_NUM_GPUS=${SLURM_NUM_GPUS:-1}"
    "SLURM_JOB_NAME=ev-ftnc-${ds}"
    "CLUSTER_MEM=${mem}"
    "CLUSTER_TIME_EVAL=$(dataset_time "${ds}")"
    "CLUSTER_CPUS=${CLUSTER_CPUS:-16}"
    "EVAL_RUN_DIR=${EVAL_RUN_DIR}"
    "CKPT=${ckpt}"
    "RUN_NAME=${ds}"
    "SIZE_TIER_WALKS=1"
    "WANDER_ADAPTIVE_WALKS=0"
    "WANDER_TEST_SAMPLES=${WANDER_TEST_SAMPLES:-16}"
    "WANDER_TRAIN_KV_CHUNK_SIZE=${WANDER_TRAIN_KV_CHUNK_SIZE:-20000}"
    "EVAL_BATCH_SIZE_NODE=${EVAL_BATCH_SIZE_NODE:-128}"
    "NC_SPLIT_INDEX=${NC_SPLIT_INDEX}"
    "IGNORE_FEATURES=${ignore_features}"
    "EVAL_SEEDS=${EVAL_SEEDS}"
  )
  if [[ -n "${WANDER_FAST_UNIFORM_WALKS:-}" ]]; then
    cmd+=("WANDER_FAST_UNIFORM_WALKS=${WANDER_FAST_UNIFORM_WALKS}")
  fi
  if [[ -n "${WANDER_COMPILE_EVAL:-}" ]]; then
    cmd+=("WANDER_COMPILE_EVAL=${WANDER_COMPILE_EVAL}")
  fi
  if [[ -n "${SIZE_TIER_WALK_NUM_DIV:-}" ]]; then
    cmd+=("SIZE_TIER_WALK_NUM_DIV=${SIZE_TIER_WALK_NUM_DIV}")
  fi
  if [[ -n "${DROP_CONSTANT_TRAIN_FEATURES:-}" ]]; then
    cmd+=("DROP_CONSTANT_TRAIN_FEATURES=${DROP_CONSTANT_TRAIN_FEATURES}")
  fi
  cmd+=(bash "${SCRIPT_DIR}/submit.sh" "${JOB_SCRIPT}" "${ds}")
  echo "    best=${grid_run} val=${score} ckpt=${ckpt}"
  printf '+ '
  printf '%q ' "${cmd[@]}"
  echo
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${cmd[@]}"
  fi
}

echo "=== NC FT test eval (best LR val ckpt, size-tier walks, split=${NC_SPLIT_INDEX}, seeds=${EVAL_SEEDS}) ==="
echo "Repo: ${REPO_ROOT}"
echo "Results: ${EVAL_RUN_DIR}"
echo "Datasets: ${DATASETS[*]}"
mkdir -p "${EVAL_RUN_DIR}"

for ds in "${DATASETS[@]}"; do
  echo "--- ${ds} (mem=$(dataset_mem "${ds}")) ---"
  if ! pick_line="$(pick_ckpt "${ds}")"; then
    echo "SKIP ${ds}: ${pick_line:-no completed LR-grid run}"
    continue
  fi
  IFS=$'\t' read -r ckpt score grid_run <<<"${pick_line}"
  submit_job "${ds}" "${ckpt}" "${score}" "${grid_run}"
done

echo "=== Done queueing (dry_run=${DRY_RUN}) ==="
echo "Monitor: squeue -u \$USER"
