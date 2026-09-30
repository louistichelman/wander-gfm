#!/usr/bin/env bash
# Login-node coordinator: zero-shot KG link-prediction eval of
# checkpoints/pretrained_wander/. Adaptive off, walk_len=128. Per-dataset
# wander_walk_num and wander_test_samples come from kg_protocol.sh (Flock n / P).
# Datasets not in that table use 128 walks and 16 samples.
# Do NOT sbatch this file.
#
# Usage (from the repo root):
#   bash slurm_scripts/eval/queue_eval_kg.sh
#   bash slurm_scripts/eval/queue_eval_kg.sh --dry-run
#   DATASETS="FB15K_237 WN18RR" bash slurm_scripts/eval/queue_eval_kg.sh
#   NO_EVALUATE_HEAD_PREDICTIONS=1 bash slurm_scripts/eval/queue_eval_kg.sh   # tails-only
#   IGNORE_EDGE_TYPES=1 DATASETS="WN18RR CONCEPTNET100K" bash slurm_scripts/eval/queue_eval_kg.sh

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: queue_eval_kg.sh is a login-node coordinator — run with bash, not sbatch." >&2
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

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/kg_protocol.sh"

# shellcheck disable=SC2206
DATASETS=(${DATASETS:-${PAPER_KG_DATASETS[*]}})

EVAL_RUN_DIR="${EVAL_RUN_DIR:-${REPO_ROOT}/results/eval/kg_$(date +%y%m%d_%H%M)}"
JOB_SCRIPT="${JOB_SCRIPT:-${SCRIPT_DIR}/eval/eval_one_dataset.sh}"

dataset_mem() {
  local ds="$1"
  case "${ds}" in
    YAGO310|DBPEDIA100K|HETIONET|CODEX_LARGE|FB_NELL|ARISTOV4|HM_INDIGO)
      echo "${LARGE_KG_MEM:-128G}" ;;
    *) echo "${CLUSTER_MEM:-64G}" ;;
  esac
}


dataset_time() {
  local ds="$1"
  case "${ds}" in
    YAGO310|DBPEDIA100K|HETIONET|CODEX_LARGE|FB_NELL|ARISTOV4|HM_INDIGO)
      echo "${CLUSTER_TIME_EVAL:-2-00:00:00}" ;;
    *) echo "${CLUSTER_TIME_EVAL:-1-00:00:00}" ;;
  esac
}

submit_job() {
  local ds="$1"
  local mem jname
  mem="$(dataset_mem "${ds}")"
  jname="${JOB_NAME_PREFIX:-ev-kg}-${ds//:/-}"
  kg_lookup_protocol "${ds}"
  local walk_num="${WANDER_WALK_NUM:-${KG_WALK_NUM}}"
  local test_samples="${WANDER_TEST_SAMPLES:-${KG_ENSEMBLE_P}}"
  local -a cmd=(
    env
    "SLURM_NUM_GPUS=${SLURM_NUM_GPUS:-1}"
    "SLURM_JOB_NAME=${jname}"
    "CLUSTER_MEM=${mem}"
    "CLUSTER_TIME_EVAL=$(dataset_time "${ds}")"
    "EVAL_RUN_DIR=${EVAL_RUN_DIR}"
    "WANDER_WALK_NUM=${walk_num}"
    "WANDER_WALK_LEN=${WANDER_WALK_LEN:-128}"
    "WANDER_ADAPTIVE_WALKS=0"
    "WANDER_TEST_SAMPLES=${test_samples}"
    "WANDER_TRAIN_KV_CHUNK_SIZE=${WANDER_TRAIN_KV_CHUNK_SIZE:-20000}"
    "NO_EVALUATE_HEAD_PREDICTIONS=${NO_EVALUATE_HEAD_PREDICTIONS:-0}"
    "IGNORE_FEATURES=${IGNORE_FEATURES:-0}"
    "IGNORE_EDGE_TYPES=${IGNORE_EDGE_TYPES:-0}"
  )
  if [[ -n "${WANDER_GLOBAL_KV_NEIGHBORS:-}" ]]; then
    cmd+=("WANDER_GLOBAL_KV_NEIGHBORS=${WANDER_GLOBAL_KV_NEIGHBORS}")
  fi
  if [[ -n "${CKPT:-}" ]]; then
    cmd+=("CKPT=${CKPT}")
  fi
  cmd+=(bash "${SCRIPT_DIR}/submit.sh" "${JOB_SCRIPT}" "${ds}")
  printf '+ '
  printf '%q ' "${cmd[@]}"
  echo
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${cmd[@]}"
  fi
}

echo "=== KG LP zero-shot eval (Flock n/P, walk_len=128, adaptive off) ==="
echo "Repo: ${REPO_ROOT}"
echo "Results: ${EVAL_RUN_DIR}"
echo "Datasets: ${DATASETS[*]}"
mkdir -p "${EVAL_RUN_DIR}"

for ds in "${DATASETS[@]}"; do
  echo "--- ${ds} (mem=$(dataset_mem "${ds}")) ---"
  kg_lookup_protocol "${ds}"
  echo "    protocol n=${WANDER_WALK_NUM:-${KG_WALK_NUM}} P=${WANDER_TEST_SAMPLES:-${KG_ENSEMBLE_P}}"
  submit_job "${ds}"
done

echo "=== Done queueing (dry_run=${DRY_RUN}) ==="
