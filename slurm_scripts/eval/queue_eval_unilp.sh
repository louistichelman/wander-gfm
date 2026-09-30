#!/usr/bin/env bash
# Login-node coordinator: UniLP sampled Recall@20 zero-shot eval of
# checkpoints/pretrained_wander/pretrained_wander.pt.
# Used by inference_ablations/queue_eval_sampled_recall.sh (paper Table 16).
# Do NOT sbatch this file.
#
# Usage (from the repo root):
#   bash slurm_scripts/eval/queue_eval_unilp.sh
#   bash slurm_scripts/eval/queue_eval_unilp.sh --dry-run
#   DATASETS="UNILP_CELEGANS UNILP_USAIR UNILP_NS" bash slurm_scripts/eval/queue_eval_unilp.sh
#   SLURM_PARTITION=gpu bash slurm_scripts/eval/queue_eval_unilp.sh

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: queue_eval_unilp.sh is a login-node coordinator — run with bash, not sbatch." >&2
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

# Paper Table 16 (sampled Recall@20, two splits) is
# eval/inference_ablations/queue_eval_sampled_recall.sh.
# shellcheck disable=SC2206
DATASETS=(${DATASETS:-${PAPER_UNILP_DATASETS[*]}})

EVAL_RUN_DIR="${EVAL_RUN_DIR:-${REPO_ROOT}/results/eval/unilp_$(date +%y%m%d_%H%M)}"
JOB_SCRIPT="${JOB_SCRIPT:-${SCRIPT_DIR}/eval/eval_one_dataset.sh}"
CKPT="${CKPT:-${REPO_ROOT}/checkpoints/pretrained_wander/pretrained_wander.pt}"
RAW_DATA_DIR="${RAW_DATA_DIR:-$(default_unilp_data_dir)}"
EVAL_SEEDS="${EVAL_SEEDS:-0:1:2:3:4:5:6}"

if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT}" >&2
  exit 1
fi
if [[ ! -d "${RAW_DATA_DIR}" ]]; then
  echo "ERROR: UniLP data dir not found: ${RAW_DATA_DIR}" >&2
  exit 1
fi

# Export once; sbatch --export=ALL inherits these (do not put lists in --export=KEY=).
export CKPT RAW_DATA_DIR EVAL_RUN_DIR EVAL_SEEDS
export IGNORE_FEATURES="${IGNORE_FEATURES:-1}"
export WANDER_WALK_NUM="${WANDER_WALK_NUM:-128}"
export WANDER_WALK_LEN="${WANDER_WALK_LEN:-128}"
export WANDER_MAX_WALK_LEN="${WANDER_MAX_WALK_LEN:-128}"
export WANDER_ADAPTIVE_WALKS="${WANDER_ADAPTIVE_WALKS:-0}"
export WANDER_TEST_SAMPLES="${WANDER_TEST_SAMPLES:-16}"
export EVAL_BATCH_SIZE_LINK=8
export BATCH_SIZE_LINK=8
export UNILP_REQUIRE_EXISTING_SPLITS="${UNILP_REQUIRE_EXISTING_SPLITS:-1}"
export LINK_PRED_EVAL="${LINK_PRED_EVAL:-sampled_recall}"
export WANDB_MODE="${WANDB_MODE:-offline}"

submit_job() {
  local ds="$1"
  local jname="unilp-${ds#UNILP_}"
  local -a cmd=(
    env
    "SLURM_NUM_GPUS=${SLURM_NUM_GPUS:-1}"
    "SLURM_JOB_NAME=${jname}"
    "CLUSTER_MEM=${CLUSTER_MEM:-64G}"
    bash "${SCRIPT_DIR}/submit.sh" "${JOB_SCRIPT}" "${ds}"
  )
  printf '+ '
  printf '%q ' "${cmd[@]}"
  echo
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${cmd[@]}"
  fi
}

echo "=== UniLP eval (protocol=${LINK_PRED_EVAL:-sampled_recall}, no features, 128x128, seeds ${EVAL_SEEDS}) ==="
echo "Repo: ${REPO_ROOT}"
echo "CKPT: ${CKPT}"
echo "Data: ${RAW_DATA_DIR}"
echo "Results: ${EVAL_RUN_DIR}"
echo "Datasets: ${DATASETS[*]}"
mkdir -p "${EVAL_RUN_DIR}"

for ds in "${DATASETS[@]}"; do
  echo "--- ${ds} ---"
  submit_job "${ds}"
done

echo "=== Done queueing (dry_run=${DRY_RUN}) ==="
