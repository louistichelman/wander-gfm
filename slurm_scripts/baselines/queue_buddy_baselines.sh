#!/usr/bin/env bash
# Login-node coordinator: one BUDDY train/eval job per AnyGraph reporting-set
# dataset. Do NOT sbatch this file.
#
# Usage (from the repo root):
#   bash slurm_scripts/baselines/queue_buddy_baselines.sh
#   bash slurm_scripts/baselines/queue_buddy_baselines.sh --dry-run
#   DATASETS="ANYGRAPH_CORA ANYGRAPH_DDI" bash slurm_scripts/baselines/queue_buddy_baselines.sh

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: queue_buddy_baselines.sh is a login-node coordinator — run with bash, not sbatch." >&2
  exit 1
fi

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" || "${1:-}" == "-n" ]]; then
  DRY_RUN=1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs checkpoints raw_data results/baselines
# shellcheck source=/dev/null
source "${SLURM_DIR}/paper_protocol.sh"

# shellcheck disable=SC2206
DATASETS=(${DATASETS:-${PAPER_LP_DATASETS[*]}})

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/baselines/buddy_$(date +%y%m%d_%H%M)}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/checkpoints/baselines/buddy}"
JOB_SCRIPT="${JOB_SCRIPT:-${SCRIPT_DIR}/run_buddy_baselines.sh}"
SEEDS="${SEEDS:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"

dataset_mem() {
  local ds="$1"
  case "${ds}" in
    ANYGRAPH_PRODUCTS_HOME|ANYGRAPH_PROTEINS_SPEC1|ANYGRAPH_SOC_EPINIONS1|ANYGRAPH_EMAIL_ENRON|ANYGRAPH_PUBMED|ANYGRAPH_CS)
      echo "${LARGE_LP_MEM:-128G}" ;;
    *) echo "${CLUSTER_MEM:-64G}" ;;
  esac
}


dataset_time() {
  local ds="$1"
  case "${ds}" in
    ANYGRAPH_PRODUCTS_HOME|ANYGRAPH_SOC_EPINIONS1|ANYGRAPH_PROTEINS_SPEC1|ANYGRAPH_EMAIL_ENRON|ANYGRAPH_PUBMED|ANYGRAPH_CS)
      echo "${CLUSTER_TIME_FINETUNE:-2-00:00:00}" ;;
    *) echo "${CLUSTER_TIME_FINETUNE:-1-00:00:00}" ;;
  esac
}

submit_job() {
  local ds="$1"
  local mem
  mem="$(dataset_mem "${ds}")"
  local -a cmd=(
    env
    "SLURM_NUM_GPUS=1"
    "SLURM_JOB_NAME=buddy-${ds}"
    "CLUSTER_MEM=${mem}"
    "CLUSTER_TIME_FINETUNE=$(dataset_time "${ds}")"
    "DATASETS=${ds}"
    "SEEDS=${SEEDS}"
    "MAX_EPOCHS=${MAX_EPOCHS}"
    "OUTPUT_DIR=${OUTPUT_DIR}"
    "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
  )
  if [[ -n "${EXTRA_CLI:-}" ]]; then
    cmd+=("EXTRA_CLI=${EXTRA_CLI}")
  fi
  if [[ -n "${FORCE_FRESH:-}" ]]; then
    cmd+=("FORCE_FRESH=${FORCE_FRESH}")
  fi
  cmd+=(bash "${SLURM_DIR}/submit.sh" "${JOB_SCRIPT}")
  printf '+ '
  printf '%q ' "${cmd[@]}"
  echo
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${cmd[@]}"
  fi
}

echo "=== BUDDY LP baseline queue (Recall@20, seed ${SEEDS}) ==="
echo "Repo: ${REPO_ROOT}"
echo "Results: ${OUTPUT_DIR}"
echo "Checkpoints: ${CHECKPOINT_DIR}"
echo "Datasets: ${DATASETS[*]}"
mkdir -p "${OUTPUT_DIR}"

for ds in "${DATASETS[@]}"; do
  echo "--- ${ds} (mem=$(dataset_mem "${ds}") time=$(dataset_time "${ds}")) ---"
  submit_job "${ds}"
done

echo "=== Done queueing (dry_run=${DRY_RUN}) ==="
