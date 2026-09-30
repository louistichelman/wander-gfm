#!/usr/bin/env bash
# Shared setup for Wander jobs. Source via source_common.sh.

set -euo pipefail

SLURM_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SLURM_SCRIPT_DIR}/_lib.sh"
# shellcheck source=/dev/null
source "${SLURM_SCRIPT_DIR}/cluster.sh"
REPO_ROOT="$(cd "${SLURM_SCRIPT_DIR}/.." && pwd)"

LOGS_DIR="${LOGS_DIR:-${REPO_ROOT}/logs}"
RAW_DATA_DIR="${RAW_DATA_DIR:-${REPO_ROOT}/raw_data}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${REPO_ROOT}/checkpoints}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results}"
CONDA_ENV="${CONDA_ENV:-wander}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_NUM_GPUS:-1}}"
PRETRAINED_CKPT="${CHECKPOINTS_DIR}/pretrained_wander/pretrained_wander.pt"

MAIN="${REPO_ROOT}/main.py"
COLLECT_EVAL_SCRIPT="${REPO_ROOT}/scripts/eval/collect_eval_run_summary.py"

# Aliases kept so existing job scripts keep working.
HOME_RAW_DATA_DIR="${RAW_DATA_DIR}"
HOME_CHECKPOINTS_DIR="${CHECKPOINTS_DIR}"
HOME_RESULTS_DIR="${RESULTS_DIR}"
JOB_SCRATCH=""

mkdir -p "${LOGS_DIR}" "${RAW_DATA_DIR}" "${CHECKPOINTS_DIR}" "${RESULTS_DIR}"

cluster_stage_datasets() { return 0; }
cluster_stage_run_dir() { return 0; }
cluster_sync_outputs() { return 0; }
cluster_resolve_path() { printf '%s' "$1"; }

cluster_default_pretrained_ckpt() {
  printf '%s' "${PRETRAINED_CKPT}"
}

cluster_latest_checkpoint() {
  local run_name="$1"
  local dir="${CHECKPOINTS_DIR}/${run_name}"
  local best="" best_epoch=-1 f base epoch
  [[ -d "${dir}" ]] || { printf '%s' ""; return 0; }
  shopt -s nullglob
  for f in "${dir}"/model_seed*_epoch*.pt; do
    [[ -f "${f}" ]] || continue
    base="$(basename "${f}")"
    if [[ "${base}" =~ epoch([0-9]+)\.pt$ ]]; then
      epoch="${BASH_REMATCH[1]}"
      if (( epoch > best_epoch )); then
        best_epoch="${epoch}"
        best="${f}"
      fi
    fi
  done
  shopt -u nullglob
  printf '%s' "${best}"
}

cluster_maybe_auto_resume() {
  local run_name="$1"
  local label="${2:-${run_name}}"

  if [[ "${FORCE_FRESH:-0}" == "1" || "${RESUME_CHECKPOINT:-}" == "none" ]]; then
    RESUME_CHECKPOINT=""
    echo "=== ${label}: starting fresh (run: ${run_name}) ==="
    return 1
  fi

  if [[ -z "${RESUME_CHECKPOINT:-}" && "${AUTO_RESUME:-1}" != "0" ]]; then
    RESUME_CHECKPOINT="$(cluster_latest_checkpoint "${run_name}")"
  fi

  if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    if [[ ! -f "${RESUME_CHECKPOINT}" ]]; then
      echo "ERROR: RESUME_CHECKPOINT not found: ${RESUME_CHECKPOINT}" >&2
      exit 1
    fi
    echo "=== ${label}: resuming from ${RESUME_CHECKPOINT} ==="
    return 0
  fi

  echo "=== ${label}: no checkpoint for ${run_name} → starting fresh ==="
  return 1
}

echo "==========================================="
echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-n/a}"
echo "Repo: ${REPO_ROOT}"
echo "==========================================="

maybe_load_conda_module
if [[ -x "${CONDA_PREFIX:-}/bin/python" ]] && \
   { [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]] || [[ "${CONDA_PREFIX}" == */"${CONDA_ENV}" ]]; }; then
  PYTHON="${CONDA_PREFIX}/bin/python"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck source=/dev/null
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
  PYTHON="${CONDA_PREFIX}/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON="$(command -v python)"
else
  echo "ERROR: no Python found. Create/activate the '${CONDA_ENV}' env (see README)." >&2
  exit 1
fi

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: missing ${PYTHON}" >&2
  exit 1
fi

export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONNOUSERSITE=1
hash -r 2>/dev/null || true

if [[ -d "${REPO_ROOT}/graph-walker/graph_walker" ]]; then
  export PYTHONPATH="${REPO_ROOT}/graph-walker${PYTHONPATH:+:${PYTHONPATH}}"
fi
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Python: ${PYTHON}"
if ! "$PYTHON" -c "import wandb" 2>/dev/null; then
  export WANDB_MODE="${WANDB_MODE:-disabled}"
fi

MASTER_PORT=$(shuf -i 29500-65535 -n 1)

run_main() {
  if [[ "${NPROC_PER_NODE}" -eq 1 ]]; then
    "$PYTHON" "${MAIN}" "$@"
  else
    "$PYTHON" -m torch.distributed.run \
      --nproc_per_node="${NPROC_PER_NODE}" \
      --master_port="${MASTER_PORT}" \
      "${MAIN}" \
      "$@"
  fi
}

finish_job() {
  echo "==========================================="
  echo "Job completed: $(date)"
  echo "==========================================="
}
