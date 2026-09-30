#!/usr/bin/env bash
#SBATCH --job-name=grids-ptrue
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/slurm_eval_grids_p_true_%x_%j.out
#SBATCH --error=logs/slurm_eval_grids_p_true_%x_%j.err

# Grids p(correct class) maps: walk-length sweep and/or train-free walk filter.
# Usage (from the repo root):
#   bash slurm_scripts/submit.sh \
#     slurm_scripts/eval/inference_ablations/eval_grids_p_true_maps.sh

set -euo pipefail

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

COLLECT_SCRIPT="${REPO_ROOT}/scripts/eval/collect_grids_p_true_maps.py"
PLOT_SCRIPT="${REPO_ROOT}/scripts/eval/plot_grids_p_true_maps.py"
if [[ ! -f "${COLLECT_SCRIPT}" ]]; then
  echo "ERROR: missing ${COLLECT_SCRIPT}" >&2
  exit 1
fi

if [[ -z "${CKPT:-}" ]]; then
  CKPT="$(cluster_default_pretrained_ckpt)"
fi
CKPT="$(cluster_resolve_path "${CKPT}")"
if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT}" >&2
  exit 1
fi

SWEEP_DIR="${SWEEP_DIR:-${HOME_RESULTS_DIR}/eval/grids_p_true_maps}"
if [[ "${SWEEP_DIR}" == "${HOME_RESULTS_DIR}"/* ]]; then
  SWEEP_DIR="${RESULTS_DIR}/${SWEEP_DIR#${HOME_RESULTS_DIR}/}"
fi

WALK_NUM="${WALK_NUM:-64}"
WALK_LEN="${WALK_LEN:-}"
WALK_LENS="${WALK_LENS:-}"
KEEP_TRAIN_FREE_P="${KEEP_TRAIN_FREE_P:-}"
KEEP_TRAIN_FREE_PS="${KEEP_TRAIN_FREE_PS:-}"
FILTER_WALK_LEN="${FILTER_WALK_LEN:-64}"
FILTER_WALK_NUM_MULT="${FILTER_WALK_NUM_MULT:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE_NODE:-128}"
TEST_SAMPLES="${WANDER_TEST_SAMPLES:-16}"
TRAIN_KV_PASSES="${WANDER_TRAIN_KV_PASSES:-16}"
SEED="${EVAL_SEED:-0}"
VIZ_GRID="${VIZ_GRID:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

mkdir -p "${SWEEP_DIR}"
cluster_stage_datasets GRIDS

cd "${REPO_ROOT}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "=== Grids p(true class) maps ==="
echo "CKPT: ${CKPT}"
echo "Output: ${SWEEP_DIR}"
echo "walk_num=${WALK_NUM} walk_len=${WALK_LEN:-} walk_lens=${WALK_LENS:-} keep_p=${KEEP_TRAIN_FREE_P:-} keep_ps=${KEEP_TRAIN_FREE_PS:-}"
echo "eval_batch=${EVAL_BATCH_SIZE} draws=${TEST_SAMPLES} filter_L=${FILTER_WALK_LEN} 2x=${FILTER_WALK_NUM_MULT}"
echo "max_radius=${NC_PROXIMITY_BATCH_MAX_RADIUS:-} batch_seeds=${PROXIMITY_BATCH_SEEDS:-0,1,2,3,4,5,6,7}"

COLLECT_ARGS=(
  --init_checkpoint "${CKPT}"
  --data_dir "${RAW_DATA_DIR}"
  --output_dir "${SWEEP_DIR}"
  --seed "${SEED}"
  --eval_batch_size "${EVAL_BATCH_SIZE}"
  --test_samples "${TEST_SAMPLES}"
  --walk_num "${WALK_NUM}"
  --filter_walk_len "${FILTER_WALK_LEN}"
  --filter_walk_num_mult "${FILTER_WALK_NUM_MULT}"
  --train_kv_passes "${TRAIN_KV_PASSES}"
)
if [[ -n "${WALK_LEN}" ]]; then
  COLLECT_ARGS+=(--walk_len "${WALK_LEN}")
fi
if [[ -n "${WALK_LENS}" ]]; then
  COLLECT_ARGS+=(--walk_lens "${WALK_LENS}")
fi
if [[ -n "${KEEP_TRAIN_FREE_P}" ]]; then
  COLLECT_ARGS+=(--keep_train_free_p "${KEEP_TRAIN_FREE_P}")
fi
if [[ -n "${KEEP_TRAIN_FREE_PS}" ]]; then
  COLLECT_ARGS+=(--keep_train_free_ps "${KEEP_TRAIN_FREE_PS}")
fi
if [[ -n "${VIZ_GRID}" ]]; then
  COLLECT_ARGS+=(--viz_grid "${VIZ_GRID}")
fi
if [[ -n "${NC_PROXIMITY_BATCH_MAX_RADIUS:-}" ]]; then
  COLLECT_ARGS+=(--nc_proximity_batch_max_radius "${NC_PROXIMITY_BATCH_MAX_RADIUS}")
fi
if [[ -n "${PROXIMITY_BATCH_SEEDS:-}" ]]; then
  COLLECT_ARGS+=(--proximity_batch_seeds "${PROXIMITY_BATCH_SEEDS}")
fi
if [[ "${SKIP_EXISTING}" == "1" || "${SKIP_EXISTING}" == "true" ]]; then
  COLLECT_ARGS+=(--skip_existing)
fi

"$PYTHON" "${COLLECT_SCRIPT}" "${COLLECT_ARGS[@]}"
"$PYTHON" "${PLOT_SCRIPT}" --sweep_dir "${SWEEP_DIR}"
echo "Maps: ${SWEEP_DIR}"
finish_job
