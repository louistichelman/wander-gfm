#!/usr/bin/env bash
# Login-node coordinator: zero-shot NC eval of checkpoints/pretrained_wander/.
# Size-tier walks (Appendix D): n<=1k 32x32, <=10k 64x128, <=40k 128x128,
# n>40k 256x128. Adaptive off. SIZE_TIER_WALK_NUM_DIV=2|4 scales K_base.
# SIZE_TIER_WALK_LEN_DIV=2|4 scales walk length only.
# Do NOT sbatch this file.
#
# Split protocol (data/nc_splits.py):
#   - several official masks -> every split, mean ± std at collect time
#   - one official split -> split 0 only
#   - no official split (GraphAny) -> 5 seeds (0..4)
#   NC_SPLITS=first  -> only split 0 (legacy)
#   EVAL_SEEDS=0:1:2 (default) — mean ± std over experiment seeds per split
#   EVAL_SEEDS=0     — single-seed (smoke / legacy)
#
# Usage (from the repo root):
#   bash slurm_scripts/eval/queue_eval_nc.sh
#   bash slurm_scripts/eval/queue_eval_nc.sh --dry-run
#   DATASETS="CORA SQUIRREL_FILTERED" bash slurm_scripts/eval/queue_eval_nc.sh
#   NC_SPLITS=first bash slurm_scripts/eval/queue_eval_nc.sh
#   EVAL_SEEDS=0 bash slurm_scripts/eval/queue_eval_nc.sh
#   IGNORE_FEATURES=1 DATASETS="PUBMED CO_CS" bash slurm_scripts/eval/queue_eval_nc.sh
#   SLURM_NUM_GPUS=2 DATASETS="HM_CATEGORIES" bash slurm_scripts/eval/queue_eval_nc.sh

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: queue_eval_nc.sh is a login-node coordinator — run with bash, not sbatch." >&2
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

# shellcheck disable=SC2206
DATASETS=(${DATASETS:-${PAPER_NC_DATASETS[*]}})
NC_SPLITS="${NC_SPLITS:-all}"
EVAL_SEEDS="${EVAL_SEEDS:-0:1:2}"

EVAL_RUN_DIR="${EVAL_RUN_DIR:-${REPO_ROOT}/results/eval/nc_$(date +%y%m%d_%H%M)}"
JOB_SCRIPT="${JOB_SCRIPT:-${SCRIPT_DIR}/eval/eval_one_dataset.sh}"

dataset_mem() {
  local ds="$1"
  case "${ds}" in
    CO_PHYSICS|HM_CATEGORIES|FULL_CORA|FULL_DBLP|POKEC_REGIONS_100K|POKEC_REGIONS_100K_TOP10|CITY_PARIS|CITY_SHANGHAI|CITY_LA)
      echo "${LARGE_NC_MEM:-256G}" ;;
    AMAZON_RATINGS|COMPUTERS|CO_CS|PUBMED|ROMAN_EMPIRE|WIKI_CS)
      echo "${MID_NC_MEM:-128G}" ;;
    *) echo "${CLUSTER_MEM:-64G}" ;;
  esac
}


dataset_time() {
  local ds="$1"
  case "${ds}" in
    CO_PHYSICS|HM_CATEGORIES|FULL_CORA|FULL_DBLP|POKEC_REGIONS_100K|POKEC_REGIONS_100K_TOP10|CITY_PARIS|CITY_SHANGHAI|CITY_LA)
      echo "${CLUSTER_TIME_EVAL:-2-00:00:00}" ;;
    *) echo "${CLUSTER_TIME_EVAL:-1-00:00:00}" ;;
  esac
}

dataset_split_indices() {
  local ds="$1"
  python3 "${REPO_ROOT}/data/nc_splits.py" "${ds}" --mode "${NC_SPLITS}"
}

submit_job() {
  local ds="$1"
  local split="$2"
  local n_splits="$3"
  local mem run_name job_name
  mem="$(dataset_mem "${ds}")"
  local ignore_features="${IGNORE_FEATURES:-0}"
  local job_prefix="ev-nc"
  if [[ "${ignore_features}" == "1" || "${ignore_features}" == "true" ]]; then
    job_prefix="ev-nf"
  fi
  if (( n_splits > 1 )); then
    run_name="${ds}/split_${split}"
    job_name="${job_prefix}-${ds}-s${split}"
  else
    run_name="${ds}"
    job_name="${job_prefix}-${ds}"
  fi
  local -a cmd=(
    env
    "SLURM_NUM_GPUS=${SLURM_NUM_GPUS:-1}"
    "SLURM_JOB_NAME=${job_name}"
    "CLUSTER_MEM=${mem}"
    "CLUSTER_TIME_EVAL=$(dataset_time "${ds}")"
    "CLUSTER_CPUS=${CLUSTER_CPUS:-16}"
    "EVAL_RUN_DIR=${EVAL_RUN_DIR}"
    "SIZE_TIER_WALKS=1"
    "WANDER_ADAPTIVE_WALKS=0"
    "WANDER_TEST_SAMPLES=${WANDER_TEST_SAMPLES:-16}"
    "WANDER_TRAIN_KV_CHUNK_SIZE=${WANDER_TRAIN_KV_CHUNK_SIZE:-20000}"
    "EVAL_BATCH_SIZE_NODE=${EVAL_BATCH_SIZE_NODE:-128}"
    "NC_SPLIT_INDEX=${split}"
    "RUN_NAME=${run_name}"
    "IGNORE_FEATURES=${ignore_features}"
    "EVAL_SEEDS=${EVAL_SEEDS}"
  )
  if [[ -n "${WANDER_FAST_UNIFORM_WALKS:-}" ]]; then
    cmd+=("WANDER_FAST_UNIFORM_WALKS=${WANDER_FAST_UNIFORM_WALKS}")
  fi
  if [[ -n "${WANDER_COMPILE_EVAL:-}" ]]; then
    cmd+=("WANDER_COMPILE_EVAL=${WANDER_COMPILE_EVAL}")
  fi
  if [[ -n "${SIZE_TIER_WALK_NUM_DIV:-}" ]]; then
    cmd+=("SIZE_TIER_WALK_NUM_DIV=${SIZE_TIER_WALK_NUM_DIV}")
  fi
  if [[ -n "${SIZE_TIER_WALK_LEN_DIV:-}" ]]; then
    cmd+=("SIZE_TIER_WALK_LEN_DIV=${SIZE_TIER_WALK_LEN_DIV}")
  fi
  if [[ -n "${WANDER_WALK_NUM:-}" ]]; then
    cmd+=("WANDER_WALK_NUM=${WANDER_WALK_NUM}")
  fi
  if [[ -n "${WANDER_EVAL_WALK_NUM:-}" ]]; then
    cmd+=("WANDER_EVAL_WALK_NUM=${WANDER_EVAL_WALK_NUM}")
  fi
  if [[ -n "${WANDER_KEEP_TRAIN_FREE_P:-}" ]]; then
    cmd+=("WANDER_KEEP_TRAIN_FREE_P=${WANDER_KEEP_TRAIN_FREE_P}")
  fi
  if [[ -n "${NC_PROXIMITY_BATCH_MAX_RADIUS:-}" ]]; then
    cmd+=("NC_PROXIMITY_BATCH_MAX_RADIUS=${NC_PROXIMITY_BATCH_MAX_RADIUS}")
  fi
  if [[ -n "${WANDER_WALK_LEN:-}" ]]; then
    cmd+=("WANDER_WALK_LEN=${WANDER_WALK_LEN}")
  fi
  if [[ -n "${WANDER_MAX_WALK_LEN:-}" ]]; then
    cmd+=("WANDER_MAX_WALK_LEN=${WANDER_MAX_WALK_LEN}")
  fi
  if [[ -n "${DROP_CONSTANT_TRAIN_FEATURES:-}" ]]; then
    cmd+=("DROP_CONSTANT_TRAIN_FEATURES=${DROP_CONSTANT_TRAIN_FEATURES}")
  fi
  if [[ -n "${CKPT:-}" ]]; then
    cmd+=("CKPT=${CKPT}")
  fi
  if [[ -n "${EVAL_EXTRA_ARGS:-}" ]]; then
    cmd+=("EVAL_EXTRA_ARGS=${EVAL_EXTRA_ARGS}")
  fi
  cmd+=(bash "${SCRIPT_DIR}/submit.sh" "${JOB_SCRIPT}" "${ds}")
  printf '+ '
  printf '%q ' "${cmd[@]}"
  echo
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${cmd[@]}"
  fi
}

echo "=== NC zero-shot eval (size-tier walks, walk_num_div=${SIZE_TIER_WALK_NUM_DIV:-1}, adaptive off, splits=${NC_SPLITS}, seeds=${EVAL_SEEDS}, ignore_features=${IGNORE_FEATURES:-0}) ==="
echo "Repo: ${REPO_ROOT}"
echo "Results: ${EVAL_RUN_DIR}"
echo "Datasets: ${DATASETS[*]}"
mkdir -p "${EVAL_RUN_DIR}"

for ds in "${DATASETS[@]}"; do
  # shellcheck disable=SC2207
  splits=( $(dataset_split_indices "${ds}") )
  n_splits="${#splits[@]}"
  echo "--- ${ds} (mem=$(dataset_mem "${ds}"), n_splits=${n_splits}: ${splits[*]}) ---"
  for split in "${splits[@]}"; do
    submit_job "${ds}" "${split}" "${n_splits}"
  done
done

echo "=== Done queueing (dry_run=${DRY_RUN}) ==="
