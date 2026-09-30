#!/usr/bin/env bash
# Login-node coordinator: zero-shot ordinary (AnyGraph) LP eval of
# checkpoints/pretrained_wander/. Recall@20, adaptive off, bs_link=8.
# Default EVAL_SEEDS=0:1:2 (frozen AnyGraph split; seeds vary walks / RNG).
# Featured graphs: 64x64 walks + PCA 64. No-feature graphs: 128x128.
# Paper Table 3 Hom-LP protocol.
# Do NOT sbatch this file.
#
# Usage (from the repo root):
#   bash slurm_scripts/eval/queue_eval_lp.sh
#   bash slurm_scripts/eval/queue_eval_lp.sh --dry-run
#   DATASETS="ANYGRAPH_CORA ANYGRAPH_DDI" bash slurm_scripts/eval/queue_eval_lp.sh
#   WANDER_WALK_NUM=64 WANDER_WALK_LEN=64 bash slurm_scripts/eval/queue_eval_lp.sh
#   EVAL_SEEDS=0 bash slurm_scripts/eval/queue_eval_lp.sh

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: queue_eval_lp.sh is a login-node coordinator — run with bash, not sbatch." >&2
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
DATASETS=(${DATASETS:-${PAPER_LP_DATASETS[*]}})
EVAL_SEEDS="${EVAL_SEEDS:-0:1:2}"

EVAL_RUN_DIR="${EVAL_RUN_DIR:-${REPO_ROOT}/results/eval/lp_$(date +%y%m%d_%H%M)}"
JOB_SCRIPT="${JOB_SCRIPT:-${SCRIPT_DIR}/eval/eval_one_dataset.sh}"

# Featured graphs: 64x64 walks + PCA 64 (adaptive off).
dataset_has_features() {
  local ds="$1"
  case "${ds}" in
    ANYGRAPH_CITESEER|ANYGRAPH_CORA|ANYGRAPH_PUBMED|ANYGRAPH_CS|ANYGRAPH_PRODUCTS_HOME)
      return 0 ;;
    *) return 1 ;;
  esac
}

dataset_walks() {
  local ds="$1"
  local scale_num="${WANDER_WALK_NUM_SCALE:-1}"
  local scale_len="${WANDER_WALK_LEN_SCALE:-1}"
  local base_num base_len num len
  if dataset_has_features "${ds}"; then
    base_num=64
    base_len=64
  else
    base_num=128
    base_len=128
  fi
  num=$((base_num / scale_num))
  len=$((base_len / scale_len))
  if (( num < 1 )); then
    num=1
  fi
  if (( len < 1 )); then
    len=1
  fi
  echo "${WANDER_WALK_NUM:-${num}} ${WANDER_WALK_LEN:-${len}}"
}

dataset_pca_link() {
  local ds="$1"
  if dataset_has_features "${ds}"; then
    echo "${PCA_TARGET_DIM_LINK:-64}"
  else
    echo "${PCA_TARGET_DIM_LINK:-32}"
  fi
}

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
      echo "${CLUSTER_TIME_EVAL:-2-00:00:00}" ;;
    *) echo "${CLUSTER_TIME_EVAL:-1-00:00:00}" ;;
  esac
}

submit_job() {
  local ds="$1"
  local mem walk_num walk_len pca_link
  mem="$(dataset_mem "${ds}")"
  read -r walk_num walk_len < <(dataset_walks "${ds}")
  pca_link="$(dataset_pca_link "${ds}")"
  local -a cmd=(
    env
    "SLURM_NUM_GPUS=${SLURM_NUM_GPUS:-1}"
    "SLURM_JOB_NAME=ev-lp-${ds}"
    "CLUSTER_MEM=${mem}"
    "CLUSTER_TIME_EVAL=$(dataset_time "${ds}")"
    "EVAL_RUN_DIR=${EVAL_RUN_DIR}"
    "BATCH_SIZE_LINK=${BATCH_SIZE_LINK:-8}"
    "EVAL_BATCH_SIZE_LINK=${EVAL_BATCH_SIZE_LINK:-8}"
    "WANDER_WALK_NUM=${walk_num}"
    "WANDER_WALK_LEN=${walk_len}"
    "WANDER_ADAPTIVE_WALKS=0"
    "WANDER_TEST_SAMPLES=${WANDER_TEST_SAMPLES:-16}"
    "WANDER_TRAIN_KV_CHUNK_SIZE=${WANDER_TRAIN_KV_CHUNK_SIZE:-20000}"
    "PCA_TARGET_DIM=${PCA_TARGET_DIM:-64}"
    "PCA_TARGET_DIM_LINK=${pca_link}"
    "ADD_INVERSE_EDGES_ANYGRAPH_BIPARTITE=${ADD_INVERSE_EDGES_ANYGRAPH_BIPARTITE:-1}"
    "EVAL_SEEDS=${EVAL_SEEDS}"
  )
  if [[ -n "${CKPT:-}" ]]; then
    cmd+=("CKPT=${CKPT}")
  fi
  if [[ -n "${WANDER_FAST_UNIFORM_WALKS:-}" ]]; then
    cmd+=("WANDER_FAST_UNIFORM_WALKS=${WANDER_FAST_UNIFORM_WALKS}")
  fi
  if [[ -n "${WANDER_COMPILE_EVAL:-}" ]]; then
    cmd+=("WANDER_COMPILE_EVAL=${WANDER_COMPILE_EVAL}")
  fi
  cmd+=(bash "${SCRIPT_DIR}/submit.sh" "${JOB_SCRIPT}" "${ds}")
  printf '+ '
  printf '%q ' "${cmd[@]}"
  echo
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${cmd[@]}"
  fi
}

echo "=== Ordinary LP zero-shot eval (Recall@20, bs_link=8, adaptive off, seeds=${EVAL_SEEDS}) ==="
echo "Repo: ${REPO_ROOT}"
echo "Results: ${EVAL_RUN_DIR}"
echo "Datasets: ${DATASETS[*]}"
echo "Featured: 64x64 + PCA 64; no-feature: 128x128"
mkdir -p "${EVAL_RUN_DIR}"

for ds in "${DATASETS[@]}"; do
  read -r walk_num walk_len < <(dataset_walks "${ds}")
  echo "--- ${ds} (walks=${walk_num}x${walk_len} pca_link=$(dataset_pca_link "${ds}") mem=$(dataset_mem "${ds}")) ---"
  submit_job "${ds}"
done

echo "=== Done queueing (dry_run=${DRY_RUN}) ==="
