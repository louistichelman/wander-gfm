#!/usr/bin/env bash
#SBATCH --job-name=wander-eval-watch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm_eval_watch_%x_%j.out
#SBATCH --error=logs/slurm_eval_watch_%x_%j.err

# Async checkpoint watcher: evaluate every new epoch checkpoint (default: Cora).
# Single-run or multi-run round-robin (one epoch per turn).
#
# Progress: checkpoints/<RUN_NAME>/eval_progress.json + wandb (wandb_run_id.txt).
#
# Submit from the repo root:
#   # Single run:
#   RUN_NAME=pt3 USE_TRAINING_ARGS_TEST_DATASETS=1 \
#     SLURM_PARTITION=gpu SLURM_NUM_GPUS=1 CLUSTER_MEM=256G \
#     bash slurm_scripts/submit.sh slurm_scripts/eval/eval_watch.sh
#
#   # Multi-run round-robin (one GPU watches several trainings):
#   RUN_NAMES=pt3,pt4 \
#     USE_TRAINING_ARGS_TEST_DATASETS=1 \
#     SLURM_PARTITION=gpu SLURM_NUM_GPUS=1 CLUSTER_MEM=256G \
#     bash slurm_scripts/submit.sh slurm_scripts/eval/eval_watch.sh
#
# Optional:
#   LATEST_ONLY=1                         # latest-only for all runs
#   LATEST_ONLY_RUNS=pt3    # per-run latest-only (comma-separated)
#   EVAL_SPLIT=test                       # epoch eval split for all runs (default: val)
#   EVAL_SPLIT_RUNS=pt3:test  # per-run split overrides (comma-separated)
#   START_EPOCH=112 POLL_INTERVAL=120
#   USE_TRAINING_ARGS_TEST_DATASETS=1     # trust each run's training_args.json

set -euo pipefail

export PYTHONUNBUFFERED=1

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
source "${_slurm}/source_common.sh"

POLL_INTERVAL="${POLL_INTERVAL:-60}"
SEED="${SEED:-0}"
START_EPOCH="${START_EPOCH:-0}"
LATEST_ONLY="${LATEST_ONLY:-0}"
USE_TRAINING_ARGS_TEST_DATASETS="${USE_TRAINING_ARGS_TEST_DATASETS:-0}"

DEFAULT_TEST_DATASETS=(CORA)

# Resolve run list: RUN_NAMES (comma-separated) wins over single RUN_NAME.
RUN_NAME_LIST=()
if [[ -n "${RUN_NAMES:-}" ]]; then
  IFS=',' read -r -a RUN_NAME_LIST <<< "${RUN_NAMES}"
  # Trim whitespace around names.
  for i in "${!RUN_NAME_LIST[@]}"; do
    RUN_NAME_LIST[$i]="$(echo "${RUN_NAME_LIST[$i]}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  done
else
  RUN_NAME_LIST=("${RUN_NAME:-main_bigger_graphs_preproc2fix}")
fi

CHECKPOINT_DIRS=()
for name in "${RUN_NAME_LIST[@]}"; do
  [[ -n "${name}" ]] || continue
  cluster_stage_run_dir "${name}"
  CHECKPOINT_DIRS+=("${CHECKPOINTS_DIR}/${name}")
done
if [[ ${#CHECKPOINT_DIRS[@]} -eq 0 ]]; then
  echo "ERROR: no RUN_NAME / RUN_NAMES provided" >&2
  exit 1
fi

# Stage default datasets unless trusting training_args (multi-run loads from each
# run's training_args; still stage the default suite as a fallback cache warm-up).
cluster_stage_datasets "${DEFAULT_TEST_DATASETS[@]}"

WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-86400}"
WAIT_POLL_SEC="${WAIT_POLL_SEC:-30}"

# For single-run, wait until training_args exists (legacy). Multi-run waits inside
# the Python watcher for each dir independently.
if [[ ${#CHECKPOINT_DIRS[@]} -eq 1 ]]; then
  CHECKPOINT_DIR="${CHECKPOINT_DIRS[0]}"
  elapsed=0
  while [[ ! -d "${CHECKPOINT_DIR}" || ! -f "${CHECKPOINT_DIR}/training_args.json" ]]; do
    if (( elapsed >= WAIT_TIMEOUT_SEC )); then
      echo "ERROR: timed out after ${WAIT_TIMEOUT_SEC}s waiting for ${CHECKPOINT_DIR}/training_args.json" >&2
      exit 1
    fi
    echo "Waiting for ${CHECKPOINT_DIR}/training_args.json (${elapsed}s)..."
    sleep "${WAIT_POLL_SEC}"
    elapsed=$((elapsed + WAIT_POLL_SEC))
  done
fi

WATCHER_ARGS=(
  --poll_interval "${POLL_INTERVAL}"
  --seed "${SEED}"
  --start_epoch "${START_EPOCH}"
)
if [[ ${#CHECKPOINT_DIRS[@]} -eq 1 ]]; then
  WATCHER_ARGS+=(--checkpoint_dir "${CHECKPOINT_DIRS[0]}")
else
  WATCHER_ARGS+=(--checkpoint_dirs "${CHECKPOINT_DIRS[@]}")
fi

if [[ -n "${MAX_EPOCH:-}" ]]; then
  WATCHER_ARGS+=(--max_epoch "${MAX_EPOCH}")
fi
if [[ "${LATEST_ONLY}" == "1" ]]; then
  WATCHER_ARGS+=(--latest_only)
fi

# Per-run latest_only: LATEST_ONLY_RUNS=name1,name2 → absolute dirs under CHECKPOINTS_DIR
if [[ -n "${LATEST_ONLY_RUNS:-}" ]]; then
  LATEST_ONLY_DIRS=()
  IFS=',' read -r -a _lor <<< "${LATEST_ONLY_RUNS}"
  for name in "${_lor[@]}"; do
    name="$(echo "${name}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [[ -n "${name}" ]] || continue
    LATEST_ONLY_DIRS+=("${CHECKPOINTS_DIR}/${name}")
  done
  if [[ ${#LATEST_ONLY_DIRS[@]} -gt 0 ]]; then
    WATCHER_ARGS+=(--latest_only_dirs "${LATEST_ONLY_DIRS[@]}")
  fi
fi

if [[ -n "${EVAL_SPLIT:-}" ]]; then
  WATCHER_ARGS+=(--eval_split "${EVAL_SPLIT}")
fi
if [[ -n "${EVAL_SPLIT_RUNS:-}" ]]; then
  EVAL_SPLIT_SPECS=()
  IFS=',' read -r -a _esr <<< "${EVAL_SPLIT_RUNS}"
  for spec in "${_esr[@]}"; do
    spec="$(echo "${spec}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [[ -n "${spec}" ]] || continue
    EVAL_SPLIT_SPECS+=("${spec}")
  done
  if [[ ${#EVAL_SPLIT_SPECS[@]} -gt 0 ]]; then
    WATCHER_ARGS+=(--eval_split_dirs "${EVAL_SPLIT_SPECS[@]}")
  fi
fi

if [[ "${USE_TRAINING_ARGS_TEST_DATASETS}" != "1" ]]; then
  WATCHER_ARGS+=(--test_datasets "${DEFAULT_TEST_DATASETS[@]}")
fi

echo "=== Checkpoint watcher ==="
echo "Runs: ${RUN_NAME_LIST[*]}"
echo "Directories: ${CHECKPOINT_DIRS[*]}"
echo "Poll interval: ${POLL_INTERVAL}s"
echo "Seed: ${SEED}"
echo "Start epoch: ${START_EPOCH}"
echo "Max epoch: ${MAX_EPOCH:-<from training_status.json>}"
echo "Latest only (all): ${LATEST_ONLY}"
echo "Latest only runs: ${LATEST_ONLY_RUNS:-<none>}"
echo "Eval split (default): ${EVAL_SPLIT:-val}"
echo "Eval split runs: ${EVAL_SPLIT_RUNS:-<none>}"
if [[ "${USE_TRAINING_ARGS_TEST_DATASETS}" == "1" ]]; then
  echo "Test datasets: from each run's training_args.json"
else
  echo "Test datasets: DEFAULT_TEST_DATASETS (${#DEFAULT_TEST_DATASETS[@]} datasets)"
fi

"$PYTHON" -m experiment.checkpoint_watcher "${WATCHER_ARGS[@]}"

finish_job
