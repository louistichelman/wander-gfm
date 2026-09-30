#!/usr/bin/env bash
# Login-node coordinator: one job per homogeneous AnyGraph dataset
# for CN / RA / PA. Do NOT sbatch this file.
#
# Usage (from the repo root):
#   bash slurm_scripts/baselines/queue_lp_heuristics.sh
#   bash slurm_scripts/baselines/queue_lp_heuristics.sh --dry-run
#   DATASETS="ANYGRAPH_CORA ANYGRAPH_PRODUCTS_HOME" bash slurm_scripts/baselines/queue_lp_heuristics.sh

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: queue_lp_heuristics.sh is a login-node coordinator — run with bash, not sbatch." >&2
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
mkdir -p logs checkpoints raw_data results/baselines/heuristics

# shellcheck disable=SC2206
DATASETS=(${DATASETS:-${PAPER_LP_DATASETS[*]}})

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/baselines/heuristics}"
JOB_SCRIPT="${JOB_SCRIPT:-${SLURM_SCRIPTS_DIR}/baselines/run_lp_heuristics.sh}"
SEEDS="${SEEDS:-0}"

dataset_mem() {
  local ds="$1"
  case "${ds}" in
    ANYGRAPH_SOC_EPINIONS1|ANYGRAPH_EMAIL_ENRON|ANYGRAPH_PROTEINS_SPEC1|ANYGRAPH_PRODUCTS_HOME|ANYGRAPH_DDI)
      echo "${LARGE_LP_MEM:-64G}" ;;
    *) echo "${CLUSTER_MEM:-32G}" ;;
  esac
}

dataset_time() {
  local ds="$1"
  case "${ds}" in
    ANYGRAPH_SOC_EPINIONS1|ANYGRAPH_PROTEINS_SPEC1)
      echo "${CLUSTER_TIME_EVAL:-0-04:00:00}" ;;
    *) echo "${CLUSTER_TIME_EVAL:-0-01:00:00}" ;;
  esac
}

export OUTPUT_DIR SEEDS
export EXTRA_CLI="${EXTRA_CLI:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"

submit_job() {
  local ds="$1"
  local jname="heur-${ds#ANYGRAPH_}"
  local -a cmd=(
    env
    "SLURM_JOB_NAME=${jname}"
    "CLUSTER_MEM=$(dataset_mem "${ds}")"
    "CLUSTER_TIME_EVAL=$(dataset_time "${ds}")"
    "OUTPUT_DIR=${OUTPUT_DIR}"
    "SEEDS=${SEEDS}"
    bash slurm_scripts/submit.sh "${JOB_SCRIPT}" "${ds}"
  )
  printf '+ '
  printf '%q ' "${cmd[@]}"
  echo
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${cmd[@]}"
  fi
}

echo "=== LP heuristics queue (CN / RA / PA, seed ${SEEDS}) ==="
echo "Repo: ${REPO_ROOT}"
echo "Results: ${OUTPUT_DIR}"
echo "Datasets: ${DATASETS[*]}"
mkdir -p "${OUTPUT_DIR}"

for ds in "${DATASETS[@]}"; do
  echo "--- ${ds} (mem=$(dataset_mem "${ds}") time=$(dataset_time "${ds}")) ---"
  submit_job "${ds}"
done

echo "=== Done queueing (dry_run=${DRY_RUN}) ==="
