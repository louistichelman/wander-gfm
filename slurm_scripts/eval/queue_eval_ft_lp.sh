#!/usr/bin/env bash
# Login-node coordinator: zero-shot test of ordinary-LP finetune epoch 5
# (completed epoch 5 → file epoch4). Same protocol as queue_eval_lp.sh:
# PCA 64, featured 64x64 + link PCA 64 / no-feature 128x128 + link PCA 32,
# adaptive off, wander_test_samples=16, uncapped Recall@20, seeds 0:1:2.
# Do NOT sbatch this file.
#
# Usage (from the repo root):
#   bash slurm_scripts/eval/queue_eval_ft_lp.sh
#   bash slurm_scripts/eval/queue_eval_ft_lp.sh --dry-run
#   DATASETS="ANYGRAPH_CORA" bash slurm_scripts/eval/queue_eval_ft_lp.sh

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: queue_eval_ft_lp.sh is a login-node coordinator — run with bash, not sbatch." >&2
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
# 1-based completed-epoch numbers; checkpoint files are 0-indexed
# (epoch 5 → model_seed0_epoch4.pt).
# shellcheck disable=SC2206
EVAL_EPOCHS=(${EVAL_EPOCHS:-5})
SEED="${SEED:-0}"
EVAL_SEEDS="${EVAL_SEEDS:-0:1:2}"
EVAL_RUN_DIR="${EVAL_RUN_DIR:-${REPO_ROOT}/results/eval/ft_lp_$(date +%y%m%d_%H%M)}"
JOB_SCRIPT="${JOB_SCRIPT:-${SCRIPT_DIR}/eval/eval_one_dataset.sh}"

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
  if dataset_has_features "${ds}"; then
    echo "${WANDER_WALK_NUM:-64} ${WANDER_WALK_LEN:-64}"
  else
    echo "${WANDER_WALK_NUM:-128} ${WANDER_WALK_LEN:-128}"
  fi
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

ckpt_path() {
  local ds="$1"
  local human_ep="$2"
  local file_ep=$((human_ep - 1))
  local run_name="${RUN_NAME_PREFIX:-ft}_${ds}_lp"
  echo "${REPO_ROOT}/checkpoints/${run_name}/model_seed${SEED}_epoch${file_ep}.pt"
}

submit_job() {
  local ds="$1"
  local human_ep="$2"
  local ckpt mem walk_num walk_len pca_link
  ckpt="$(ckpt_path "${ds}" "${human_ep}")"
  if [[ ! -f "${ckpt}" && "${DRY_RUN}" -eq 0 ]]; then
    echo "SKIP ${ds} ep${human_ep}: missing ${ckpt}"
    return 0
  fi
  if [[ ! -f "${ckpt}" ]]; then
    echo "WARN ${ds} ep${human_ep}: checkpoint not on disk yet: ${ckpt}"
  fi
  mem="$(dataset_mem "${ds}")"
  read -r walk_num walk_len < <(dataset_walks "${ds}")
  pca_link="$(dataset_pca_link "${ds}")"
  local run_name="${ds}_ep${human_ep}"
  local -a cmd=(
    env
    "SLURM_NUM_GPUS=${SLURM_NUM_GPUS:-1}"
    "SLURM_JOB_NAME=ev-ftlp-${ds}-ep${human_ep}"
    "CLUSTER_MEM=${mem}"
    "CLUSTER_TIME_EVAL=$(dataset_time "${ds}")"
    "EVAL_RUN_DIR=${EVAL_RUN_DIR}"
    "CKPT=${ckpt}"
    "RUN_NAME=${run_name}"
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
    bash "${SCRIPT_DIR}/submit.sh" "${JOB_SCRIPT}" "${ds}"
  )
  printf '+ '
  printf '%q ' "${cmd[@]}"
  echo
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${cmd[@]}"
  fi
}

echo "=== Ordinary LP FT test eval (epochs ${EVAL_EPOCHS[*]}, ZS PCA 64, seeds=${EVAL_SEEDS}, ts=16) ==="
echo "Repo: ${REPO_ROOT}"
echo "Results: ${EVAL_RUN_DIR}"
echo "Datasets: ${DATASETS[*]}"
echo "Featured: 64x64 + PCA 64; no-feature: 128x128; epoch 5 → epoch4.pt"
mkdir -p "${EVAL_RUN_DIR}"

for ds in "${DATASETS[@]}"; do
  for ep in "${EVAL_EPOCHS[@]}"; do
    if [[ ! "${ep}" =~ ^[1-9][0-9]*$ ]]; then
      echo "ERROR: EVAL_EPOCHS must be 1-based completed epoch numbers (got ${ep})" >&2
      exit 1
    fi
    echo "--- ${ds} ep${ep} ckpt=$(ckpt_path "${ds}" "${ep}") ---"
    submit_job "${ds}" "${ep}"
  done
done

echo "=== Done queueing (dry_run=${DRY_RUN}) ==="
echo "Monitor: squeue -u \$USER"
