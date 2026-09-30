#!/usr/bin/env bash
# Login-node coordinator: one job per (method, dataset) for the paper
# Hom-LP grids (BUDDY 12, NBFNet 18). Do NOT sbatch this file.
#
# Usage (from the repo root):
#   bash slurm_scripts/baselines/queue_lp_grid.sh
#   bash slurm_scripts/baselines/queue_lp_grid.sh --dry-run
#   METHODS="buddy" DATASETS="ANYGRAPH_CORA ANYGRAPH_CITESEER" bash slurm_scripts/baselines/queue_lp_grid.sh
#   METHODS="nbfnet" DATASETS="ANYGRAPH_CORA" bash slurm_scripts/baselines/queue_lp_grid.sh

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: queue_lp_grid.sh is a login-node coordinator — run with bash, not sbatch." >&2
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
mkdir -p logs checkpoints raw_data results/baselines/lp_grid

LP_METHODS=(buddy nbfnet)
# shellcheck disable=SC2206
METHODS=(${METHODS:-${LP_METHODS[*]}})
# shellcheck disable=SC2206
DATASETS=(${DATASETS:-${PAPER_LP_DATASETS[*]}})

JOB_SCRIPT="${JOB_SCRIPT:-${SLURM_SCRIPTS_DIR}/baselines/run_lp_grid.sh}"

dataset_mem() {
  local ds="$1"
  case "${ds}" in
    ANYGRAPH_SOC_EPINIONS1|ANYGRAPH_EMAIL_ENRON|ANYGRAPH_PROTEINS_SPEC1|ANYGRAPH_PRODUCTS_HOME)
      echo "${LARGE_LP_MEM:-128G}" ;;
    *) echo "${CLUSTER_MEM:-64G}" ;;
  esac
}

dataset_time() {
  local method="$1"
  local ds="$2"
  if [[ "${method}" == "nbfnet" ]]; then
    case "${ds}" in
      ANYGRAPH_SOC_EPINIONS1)
        echo "${CLUSTER_TIME_TRAIN:-7-00:00:00}" ;;
      ANYGRAPH_EMAIL_ENRON|ANYGRAPH_PROTEINS_SPEC1|ANYGRAPH_DDI|ANYGRAPH_PUBMED|ANYGRAPH_PRODUCTS_HOME)
        echo "${CLUSTER_TIME_FINETUNE:-5-00:00:00}" ;;
      *) echo "${CLUSTER_TIME_FINETUNE:-3-00:00:00}" ;;
    esac
  else
    case "${ds}" in
      ANYGRAPH_SOC_EPINIONS1|ANYGRAPH_EMAIL_ENRON|ANYGRAPH_PROTEINS_SPEC1|ANYGRAPH_DDI)
        echo "${CLUSTER_TIME_FINETUNE:-4-00:00:00}" ;;
      *) echo "${CLUSTER_TIME_FINETUNE:-2-00:00:00}" ;;
    esac
  fi
}

export FORCE_FRESH="${FORCE_FRESH:-0}"
export EXTRA_CLI="${EXTRA_CLI:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"

submit_job() {
  local method="$1"
  local ds="$2"
  local grid_subdir="${LP_GRID_SUBDIR:-${method}}"
  local jname="lpgrid-${grid_subdir}-${ds#ANYGRAPH_}"
  local out_dir="${REPO_ROOT}/results/baselines/lp_grid/${grid_subdir}"
  local ckpt_dir="${REPO_ROOT}/checkpoints/baselines/lp_grid/${grid_subdir}"
  local -a cmd=(
    env
    "SLURM_JOB_NAME=${jname}"
    "CLUSTER_MEM=$(dataset_mem "${ds}")"
    "CLUSTER_TIME_TRAIN=$(dataset_time "${method}" "${ds}")"
    "CLUSTER_TIME_FINETUNE=$(dataset_time "${method}" "${ds}")"
    "OUTPUT_DIR=${out_dir}"
    "CHECKPOINT_DIR=${ckpt_dir}"
    bash slurm_scripts/submit.sh "${JOB_SCRIPT}" "${method}" "${ds}"
  )
  printf '+ '
  printf '%q ' "${cmd[@]}"
  echo
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${cmd[@]}"
  fi
}

echo "=== LP baseline grid (BUDDY 12 / NBFNet 18, NBFNet max 40 ep) ==="
echo "Repo: ${REPO_ROOT}"
echo "Methods: ${METHODS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "LP_GRID_SUBDIR: ${LP_GRID_SUBDIR:-<method>}"
echo "EXTRA_CLI: ${EXTRA_CLI:-}"

for method in "${METHODS[@]}"; do
  grid_subdir="${LP_GRID_SUBDIR:-${method}}"
  mkdir -p "${REPO_ROOT}/results/baselines/lp_grid/${grid_subdir}" \
    "${REPO_ROOT}/checkpoints/baselines/lp_grid/${grid_subdir}"
  for ds in "${DATASETS[@]}"; do
    echo "--- ${method} ${ds} (mem=$(dataset_mem "${ds}") time=$(dataset_time "${method}" "${ds}")) ---"
    submit_job "${method}" "${ds}"
  done
done

echo "=== Done queueing (dry_run=${DRY_RUN}) ==="
