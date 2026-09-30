#!/usr/bin/env bash
# Login-node coordinator: Figure 2 Grids p(true class) maps.
#
# Eval batch size 128, default walk_num=64 (size-tier for Grids).
#   1) walk_len in {4,8,16,32,64,128}
#   2) walk_len=64, 2x walks, keep train-free walks with p in {0,0.1,0.5}
#
# Usage (from the repo root):
#   bash slurm_scripts/eval/inference_ablations/queue_eval_grids_p_true_maps.sh
#   bash slurm_scripts/eval/inference_ablations/queue_eval_grids_p_true_maps.sh --dry-run

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: queue_eval_grids_p_true_maps.sh is a login-node coordinator." >&2
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

JOB_SCRIPT="${SCRIPT_DIR}/eval/inference_ablations/eval_grids_p_true_maps.sh"
PLOT_SCRIPT="${SCRIPT_DIR}/eval/inference_ablations/plot_grids_p_true_maps.sh"
CKPT="${CKPT:-${REPO_ROOT}/checkpoints/pretrained_wander/pretrained_wander.pt}"
SWEEP_DIR="${SWEEP_DIR:-${REPO_ROOT}/results/eval/grids_p_true_maps_$(date +%y%m%d_%H%M)}"
mkdir -p "${SWEEP_DIR}" logs/grids_p_true_maps

if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT}" >&2
  exit 1
fi

export CKPT
export SWEEP_DIR
export WANDB_MODE="${WANDB_MODE:-offline}"
export WALK_NUM="${WALK_NUM:-64}"
export FILTER_WALK_LEN="${FILTER_WALK_LEN:-64}"
export FILTER_WALK_NUM_MULT="${FILTER_WALK_NUM_MULT:-2}"
export EVAL_BATCH_SIZE_NODE="${EVAL_BATCH_SIZE_NODE:-128}"
export WANDER_TEST_SAMPLES="${WANDER_TEST_SAMPLES:-16}"
export WANDER_TRAIN_KV_PASSES="${WANDER_TRAIN_KV_PASSES:-16}"
export EVAL_SEED="${EVAL_SEED:-0}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"
export VIZ_GRID="${VIZ_GRID:-}"
export NC_PROXIMITY_BATCH_MAX_RADIUS="${NC_PROXIMITY_BATCH_MAX_RADIUS:-}"
export PROXIMITY_BATCH_SEEDS="${PROXIMITY_BATCH_SEEDS:-0,1,2,3,4,5,6,7}"

WALK_LENS=(${WALK_LENS:-4 8 16 32 64 128})
KEEP_PS=(${KEEP_PS:-0 0.1 0.5})

echo "=== Grids p(true class) maps ==="
echo "Repo: ${REPO_ROOT}"
echo "CKPT: ${CKPT}"
echo "Results: ${SWEEP_DIR}"
echo "walk_num=${WALK_NUM} lens=${WALK_LENS[*]} keep_p=${KEEP_PS[*]} batch=${EVAL_BATCH_SIZE_NODE} max_radius=${NC_PROXIMITY_BATCH_MAX_RADIUS:-} batch_seeds=${PROXIMITY_BATCH_SEEDS}"

MANIFEST="${SWEEP_DIR}/sweep_manifest.tsv"
printf 'kind\twalk_num\twalk_len\tkeep_p\trun_name\tjob_id\n' > "${MANIFEST}"

N_SUBMIT=0
N_SKIP=0
JOB_IDS=()

submit_cell() {
  local kind="$1" n="$2" l="$3" p="$4" run_name="$5" jname="$6"
  local maps="${SWEEP_DIR}/${run_name}/maps.npz"
  if [[ "${SKIP_EXISTING}" == "1" && -s "${maps}" ]]; then
    echo "  skip ${run_name} (exists)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${kind}" "${n}" "${l}" "${p}" "${run_name}" "skip" >> "${MANIFEST}"
    N_SKIP=$((N_SKIP + 1))
    return 0
  fi
  echo "  submit ${run_name}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${kind}" "${n}" "${l}" "${p}" "${run_name}" "dry-run" >> "${MANIFEST}"
    N_SUBMIT=$((N_SUBMIT + 1))
    return 0
  fi
  local submit_log
  submit_log="$(mktemp)"
  local env_args=(
    SLURM_PARTITION="${SLURM_PARTITION:-gpu}"
    SLURM_NUM_GPUS=1
    SLURM_JOB_NAME="${jname}"
    CLUSTER_MEM=64G
    CLUSTER_TIME_EVAL=0-08:00:00
    CLUSTER_CPUS=8
    CKPT="${CKPT}"
    SWEEP_DIR="${SWEEP_DIR}"
    WALK_NUM="${WALK_NUM}"
    FILTER_WALK_LEN="${FILTER_WALK_LEN}"
    FILTER_WALK_NUM_MULT="${FILTER_WALK_NUM_MULT}"
    EVAL_BATCH_SIZE_NODE="${EVAL_BATCH_SIZE_NODE}"
    WANDER_TEST_SAMPLES="${WANDER_TEST_SAMPLES}"
    WANDER_TRAIN_KV_PASSES="${WANDER_TRAIN_KV_PASSES}"
    EVAL_SEED="${EVAL_SEED}"
    SKIP_EXISTING="${SKIP_EXISTING}"
    PROXIMITY_BATCH_SEEDS="${PROXIMITY_BATCH_SEEDS}"
  )
  if [[ -n "${NC_PROXIMITY_BATCH_MAX_RADIUS}" ]]; then
    env_args+=(NC_PROXIMITY_BATCH_MAX_RADIUS="${NC_PROXIMITY_BATCH_MAX_RADIUS}")
  fi
  if [[ -n "${VIZ_GRID}" ]]; then
    env_args+=(VIZ_GRID="${VIZ_GRID}")
  fi
  if [[ "${kind}" == "walk_len" ]]; then
    env_args+=(WALK_LEN="${l}" KEEP_TRAIN_FREE_P="" KEEP_TRAIN_FREE_PS="" WALK_LENS="")
  else
    env_args+=(WALK_LEN="" WALK_LENS="" KEEP_TRAIN_FREE_P="${p}" KEEP_TRAIN_FREE_PS="")
  fi
  env "${env_args[@]}" bash "${SCRIPT_DIR}/submit.sh" "${JOB_SCRIPT}" \
    | tee "${submit_log}"
  local job_id
  job_id="$(awk '/Submitted batch job/ { print $NF; exit }' "${submit_log}")"
  rm -f "${submit_log}"
  if [[ -z "${job_id}" ]]; then
    echo "ERROR: failed to submit ${run_name}" >&2
    return 1
  fi
  JOB_IDS+=("${job_id}")
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${kind}" "${n}" "${l}" "${p}" "${run_name}" "${job_id}" >> "${MANIFEST}"
  N_SUBMIT=$((N_SUBMIT + 1))
}

echo "--- walk_len cells ---"
for l in "${WALK_LENS[@]}"; do
  run_name="n${WALK_NUM}_l${l}"
  submit_cell "walk_len" "${WALK_NUM}" "${l}" "-" "${run_name}" "ev-gr-pt-l${l}"
done

echo "--- train-free filter cells ---"
filter_n=$((WALK_NUM * FILTER_WALK_NUM_MULT))
for p in "${KEEP_PS[@]}"; do
  if [[ "${p}" == "0" || "${p}" == "0.0" ]]; then
    run_name="n${filter_n}_l${FILTER_WALK_LEN}_p0"
  else
    run_name="n${filter_n}_l${FILTER_WALK_LEN}_p${p}"
  fi
  jtag="${p//./}"
  submit_cell "filter" "${filter_n}" "${FILTER_WALK_LEN}" "${p}" "${run_name}" "ev-gr-pt-p${jtag}"
done

echo "--- plot (afterany) ---"
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "  dry-run plot"
else
  DEP=""
  if [[ ${#JOB_IDS[@]} -gt 0 ]]; then
    DEP="$(IFS=:; echo "${JOB_IDS[*]}")"
  fi
  PLOT_CMD=(
    env
    SLURM_PARTITION="${SLURM_PARTITION:-gpu}"
    SLURM_NUM_GPUS=1
    SLURM_JOB_NAME="grids-ptrue-plot"
    CLUSTER_MEM=16G
    CLUSTER_TIME_EVAL=0-00:30:00
    CLUSTER_CPUS=4
    SWEEP_DIR="${SWEEP_DIR}"
  )
  if [[ -n "${DEP}" ]]; then
    PLOT_CMD+=("SLURM_DEPENDENCY=afterany:${DEP}")
  fi
  PLOT_CMD+=(bash "${SCRIPT_DIR}/submit.sh" "${PLOT_SCRIPT}")
  "${PLOT_CMD[@]}"
fi

echo "=== Done queueing (dry_run=${DRY_RUN} submitted=${N_SUBMIT} skipped=${N_SKIP}) ==="
echo "Manifest: ${MANIFEST}"
echo "Results: ${SWEEP_DIR}"
