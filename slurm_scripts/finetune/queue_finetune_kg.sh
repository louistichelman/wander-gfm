#!/usr/bin/env bash
# Login-node coordinator: queue KG LP finetunes from
# checkpoints/pretrained_wander/. Paper protocol: lr=5e-5, wd=0, 256 negatives,
# full training split each epoch, up to 5 epochs, best validation MRR.
# Walks n from kg_protocol.sh. --skip_final_test; after FT:
#   bash slurm_scripts/eval/queue_eval_ft_kg.sh
# Do NOT sbatch this file.
#
# Usage (from the repo root):
#   bash slurm_scripts/finetune/queue_finetune_kg.sh
#   bash slurm_scripts/finetune/queue_finetune_kg.sh --dry-run
#   DATASETS="FB15K_237 WN18RR" bash slurm_scripts/finetune/queue_finetune_kg.sh

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: queue_finetune_kg.sh is a login-node coordinator — run with bash, not sbatch." >&2
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

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/kg_protocol.sh"

# shellcheck disable=SC2206
DATASETS=(${DATASETS:-${PAPER_KG_DATASETS[*]}})
FT_GPUS="${FT_GPUS:-1}"

dataset_mem() {
  local ds="$1"
  case "${ds}" in
    YAGO310|DBPEDIA100K|HETIONET|CODEX_LARGE|FB_NELL|ARISTOV4|HM_INDIGO)
      echo "${LARGE_KG_MEM:-128G}" ;;
    *) echo "${CLUSTER_MEM:-64G}" ;;
  esac
}

submit_job() {
  local ds="$1"
  local mem jname
  mem="$(dataset_mem "${ds}")"
  jname="ft-kg-${ds//:/-}"
  kg_lookup_protocol "${ds}"
  local -a cmd=(
    env
    "SLURM_NUM_GPUS=${FT_GPUS}"
    "SLURM_JOB_NAME=${jname}"
    "CLUSTER_MEM=${mem}"
    "CLUSTER_TIME_FINETUNE=${CLUSTER_TIME_FINETUNE:-2-00:00:00}"
    "RUN_NAME=${RUN_NAME_PREFIX:-ft}_${ds}_kg"
    "FORCE_FRESH=${FORCE_FRESH:-0}"
  )
  if [[ -n "${MAX_EPOCHS:-}" ]]; then
    cmd+=("MAX_EPOCHS=${MAX_EPOCHS}")
  fi
  if [[ -n "${PATIENCE:-}" ]]; then
    cmd+=("PATIENCE=${PATIENCE}")
  fi
  if [[ -n "${BATCH_PER_EPOCH:-}" ]]; then
    cmd+=("BATCH_PER_EPOCH=${BATCH_PER_EPOCH}")
  fi
  cmd+=(
    bash "${SCRIPT_DIR}/submit.sh" "${SCRIPT_DIR}/finetune/finetune_kg.sh" "${ds}"
  )
  printf '+ '
  printf '%q ' "${cmd[@]}"
  echo
  echo "    Flock n=${KG_WALK_NUM} bs=${KG_BATCH_SIZE} bpe=full epochs=5 val_P=2 kv_nbrs=200 (best val MRR)"
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${cmd[@]}"
  fi
}

echo "=== KG LP finetune (5 full epochs, best val MRR, skip_final_test) ==="
echo "Repo: ${REPO_ROOT}"
echo "Datasets: ${DATASETS[*]}"
echo "After FT: bash slurm_scripts/eval/queue_eval_ft_kg.sh"

for ds in "${DATASETS[@]}"; do
  echo "--- ft_${ds}_kg mem=$(dataset_mem "${ds}") ---"
  submit_job "${ds}"
done

echo "=== Done queueing (dry_run=${DRY_RUN}) ==="
echo "Monitor: squeue -u \$USER"
