#!/usr/bin/env bash
# Login-node coordinator: zero-shot test of KG finetune best checkpoints
# (model_seed0_best.pt). Same Flock n/P and walk_len=128 as queue_eval_kg.sh.
# Do NOT sbatch this file.
#
# Usage (from the repo root):
#   bash slurm_scripts/eval/queue_eval_ft_kg.sh
#   bash slurm_scripts/eval/queue_eval_ft_kg.sh --dry-run
#   DATASETS="FB15K_237 WN18RR" bash slurm_scripts/eval/queue_eval_ft_kg.sh

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: queue_eval_ft_kg.sh is a login-node coordinator — run with bash, not sbatch." >&2
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
SEED="${SEED:-0}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-ft}"

EVAL_RUN_DIR="${EVAL_RUN_DIR:-${REPO_ROOT}/results/eval/ft_kg_$(date +%y%m%d_%H%M)}"
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

ckpt_path() {
  local ds="$1"
  echo "${REPO_ROOT}/checkpoints/${RUN_NAME_PREFIX}_${ds}_kg/model_seed${SEED}_best.pt"
}

submit_job() {
  local ds="$1"
  local ckpt mem jname
  ckpt="$(ckpt_path "${ds}")"
  if [[ ! -f "${ckpt}" && "${DRY_RUN}" -eq 0 ]]; then
    echo "SKIP ${ds}: missing ${ckpt}"
    return 0
  fi
  if [[ ! -f "${ckpt}" ]]; then
    echo "WARN ${ds}: checkpoint not on disk yet: ${ckpt}"
  fi
  mem="$(dataset_mem "${ds}")"
  jname="ev-ftkg-${ds//:/-}"
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
    "CKPT=${ckpt}"
    "RUN_NAME=${ds}"
    "WANDER_WALK_NUM=${walk_num}"
    "WANDER_WALK_LEN=${WANDER_WALK_LEN:-128}"
    "WANDER_ADAPTIVE_WALKS=0"
    "WANDER_TEST_SAMPLES=${test_samples}"
    "WANDER_TRAIN_KV_CHUNK_SIZE=${WANDER_TRAIN_KV_CHUNK_SIZE:-20000}"
    "NO_EVALUATE_HEAD_PREDICTIONS=${NO_EVALUATE_HEAD_PREDICTIONS:-0}"
  )
  if [[ -n "${WANDER_GLOBAL_KV_NEIGHBORS:-}" ]]; then
    cmd+=("WANDER_GLOBAL_KV_NEIGHBORS=${WANDER_GLOBAL_KV_NEIGHBORS}")
  fi
  cmd+=(bash "${SCRIPT_DIR}/submit.sh" "${JOB_SCRIPT}" "${ds}")
  printf '+ '
  printf '%q ' "${cmd[@]}"
  echo
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${cmd[@]}"
  fi
}

echo "=== KG FT test eval (best val MRR ckpt, Flock n/P, walk_len=128) ==="
echo "Repo: ${REPO_ROOT}"
echo "Results: ${EVAL_RUN_DIR}"
echo "Datasets: ${DATASETS[*]}"
mkdir -p "${EVAL_RUN_DIR}"

for ds in "${DATASETS[@]}"; do
  echo "--- ${ds} ckpt=$(ckpt_path "${ds}") ---"
  kg_lookup_protocol "${ds}"
  echo "    protocol n=${WANDER_WALK_NUM:-${KG_WALK_NUM}} P=${WANDER_TEST_SAMPLES:-${KG_ENSEMBLE_P}}"
  submit_job "${ds}"
done

echo "=== Done queueing (dry_run=${DRY_RUN}) ==="
echo "Monitor: squeue -u \$USER"
