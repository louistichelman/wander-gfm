#!/usr/bin/env bash
# Login-node coordinator: CYCLES finetune (appendix).
# Wander: 300 epochs, 32 BPE, train/eval batch 16, walks 32×16, val-selected
# checkpoint; remaining hparams match pretraining (lr=5e-5, wd=0.01).
# Optionally queues official GraphPFN finetune and the GCN grid.
# Do NOT sbatch this file.
#
# Usage (from the repo root):
#   bash slurm_scripts/finetune/queue_finetune_cycles.sh
#   bash slurm_scripts/finetune/queue_finetune_cycles.sh --dry-run
#   INCLUDE_BASELINES=0 bash slurm_scripts/finetune/queue_finetune_cycles.sh

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: queue_finetune_cycles.sh is a login-node coordinator — run with bash, not sbatch." >&2
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
SCRIPT_DIR="${SLURM_SCRIPTS_DIR}"
cd "${REPO_ROOT}"
mkdir -p logs checkpoints raw_data

INCLUDE_BASELINES="${INCLUDE_BASELINES:-1}"
RUN_NAME="${RUN_NAME:-ft_CYCLES_paper}"
JOB_SCRIPT="${SCRIPT_DIR}/finetune/finetune_nc.sh"

echo "=== CYCLES finetune ==="
echo "Repo: ${REPO_ROOT}"
echo "Wander: 300 ep, 32 BPE, train/eval bs=16, walks 32x16, lr=${LR:-5e-5}, wd=${WEIGHT_DECAY:-0.01}"
echo "Baselines: ${INCLUDE_BASELINES}"

wander_cmd=(
  env
  "SLURM_NUM_GPUS=${FT_GPUS:-1}"
  "SLURM_JOB_NAME=ft-cycles"
  "CLUSTER_MEM=${CLUSTER_MEM:-64G}"
  "CLUSTER_TIME_FINETUNE=${CLUSTER_TIME_FINETUNE:-2-00:00:00}"
  "RUN_NAME=${RUN_NAME}"
  "MAX_EPOCHS=${MAX_EPOCHS:-300}"
  "PATIENCE=${PATIENCE:-300}"
  "BATCH_PER_EPOCH=${BATCH_PER_EPOCH:-32}"
  "BATCH_SIZE_NODE=${BATCH_SIZE_NODE:-16}"
  "EVAL_BATCH_SIZE_NODE=${EVAL_BATCH_SIZE_NODE:-16}"
  "LR=${LR:-5e-5}"
  "WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}"
  "WANDER_WALK_NUM=${WANDER_WALK_NUM:-32}"
  "WANDER_WALK_LEN=${WANDER_WALK_LEN:-16}"
  "WANDER_MAX_WALK_LEN=${WANDER_MAX_WALK_LEN:-128}"
  "WANDER_ADAPTIVE_WALKS=0"
  "NC_TRAIN_QUERY_FRAC=${NC_TRAIN_QUERY_FRAC:-0}"
  "NC_TRAIN_QUERY_CAP=${NC_TRAIN_QUERY_CAP:-0}"
  "IGNORE_FEATURES=1"
  "SKIP_FINAL_TEST=${SKIP_FINAL_TEST:-0}"
  "FORCE_FRESH=${FORCE_FRESH:-0}"
  bash "${SCRIPT_DIR}/submit.sh" "${JOB_SCRIPT}" CYCLES
)
printf '+ '
printf '%q ' "${wander_cmd[@]}"
echo
if [[ "${DRY_RUN}" -eq 0 ]]; then
  "${wander_cmd[@]}"
fi

if [[ "${INCLUDE_BASELINES}" == "1" || "${INCLUDE_BASELINES}" == "true" ]]; then
  echo "=== GraphPFN official finetune (cycles, nofeat) ==="
  SUBMIT_PYTHON="$(require_env_python "${WANDER_ENV:-wander}" "${SUBMIT_PYTHON:-}")"
  export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  graphpfn_gen=(
    "${SUBMIT_PYTHON}" baselines/graphpfn/generate_finetune_configs.py --nofeat --datasets cycles
  )
  printf '+ '
  printf '%q ' "${graphpfn_gen[@]}"
  echo
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${graphpfn_gen[@]}"
  fi
  graphpfn_cmd=(
    env
    "GRAPHPFN_DATASET=cycles"
    "SLURM_NUM_GPUS=1"
    "SLURM_JOB_NAME=graphpfn-ft-cycles"
    "CLUSTER_MEM=${CLUSTER_MEM:-64G}"
    bash "${SCRIPT_DIR}/submit.sh" "${SCRIPT_DIR}/baselines/graphpfn_finetune.sh"
  )
  printf '+ '
  printf '%q ' "${graphpfn_cmd[@]}"
  echo
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${graphpfn_cmd[@]}"
  fi

  echo "=== GCN grid (CYCLES, nofeat) ==="
  gcn_cmd=(
    env
    "DATASETS=CYCLES"
    "GCN_GRID_IGNORE_FEATURES=1"
    bash "${SCRIPT_DIR}/baselines/queue_gcn_grid.sh"
  )
  printf '+ '
  printf '%q ' "${gcn_cmd[@]}"
  echo
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${gcn_cmd[@]}"
  fi
fi

echo "=== Done queueing CYCLES (dry_run=${DRY_RUN}) ==="
