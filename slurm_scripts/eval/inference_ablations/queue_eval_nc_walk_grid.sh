#!/usr/bin/env bash
# Login-node coordinator: Wander NC walk grid on the paper NC reporting set.
#
# Grid (eval_batch_size_node=128, adaptive off, otherwise queue_eval_nc.sh):
#   wander_eval_walk_num  in {64, 128}  (+ 32 if n<1k)
#   wander_walk_len       in {32, 64, 128}
#   wander_train_kv_walk_num:
#     n<1k              -> {32, 64, 128}
#     n_train<1k        -> {64, 128}
#     else              -> {128, 256, 512}
#
# Usage (from the repo root):
#   bash slurm_scripts/eval/inference_ablations/queue_eval_nc_walk_grid.sh
#   bash slurm_scripts/eval/inference_ablations/queue_eval_nc_walk_grid.sh --dry-run

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: queue_eval_nc_walk_grid.sh is a login-node coordinator." >&2
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

JOB_SCRIPT="${JOB_SCRIPT:-${SCRIPT_DIR}/eval/eval_one_dataset.sh}"
CKPT="${CKPT:-${REPO_ROOT}/checkpoints/pretrained_wander/pretrained_wander.pt}"
RAW_DATA_DIR="${RAW_DATA_DIR:-${REPO_ROOT}/raw_data}"
PYTHON="${PYTHON:-python3}"

# shellcheck disable=SC2206
DATASETS=(${DATASETS:-${PAPER_NC_DATASETS[*]}})

GRID_ROOT="${NC_WALK_GRID_DIR:-${REPO_ROOT}/results/eval/nc_walk_grid_$(date +%y%m%d_%H%M)}"
mkdir -p "${GRID_ROOT}" logs/nc_walk_grid

if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: python not executable: ${PYTHON}" >&2
  exit 1
fi

dataset_mem() {
  local ds="$1"
  case "${ds}" in
    CO_PHYSICS|HM_CATEGORIES|FULL_CORA|FULL_DBLP|POKEC_REGIONS_100K|POKEC_REGIONS_100K_TOP10|CITY_PARIS|CITY_SHANGHAI|CITY_LA)
      echo "${LARGE_NC_MEM:-256G}" ;;
    AMAZON_RATINGS|COMPUTERS|CO_CS|PUBMED|ROMAN_EMPIRE|WIKI_CS|ACTOR)
      echo "${MID_NC_MEM:-128G}" ;;
    *) echo "${CLUSTER_MEM:-64G}" ;;
  esac
}

dataset_time() {
  local ds="$1"
  case "${ds}" in
    CO_PHYSICS|HM_CATEGORIES|FULL_CORA|FULL_DBLP|POKEC_REGIONS_100K|POKEC_REGIONS_100K_TOP10|CITY_PARIS|CITY_SHANGHAI|CITY_LA)
      echo "${CLUSTER_TIME_EVAL:-2-00:00:00}" ;;
    AMAZON_RATINGS|COMPUTERS|CO_CS|PUBMED|ROMAN_EMPIRE|WIKI_CS|ACTOR|FULL_DBLP)
      echo "${CLUSTER_TIME_EVAL:-1-00:00:00}" ;;
    *) echo "${CLUSTER_TIME_EVAL:-0-12:00:00}" ;;
  esac
}

# Shared NC protocol (queue_eval_nc.sh). Per-cell walk vars are set in submit_cell.
export CKPT RAW_DATA_DIR
export IGNORE_FEATURES=0
export EVAL_SEEDS=0
export WANDER_ADAPTIVE_WALKS=0
export WANDER_TEST_SAMPLES="${WANDER_TEST_SAMPLES:-16}"
export WANDER_TRAIN_KV_CHUNK_SIZE="${WANDER_TRAIN_KV_CHUNK_SIZE:-20000}"
export EVAL_BATCH_SIZE_NODE=128
export BATCH_SIZE_LINK=4
export EVAL_BATCH_SIZE_LINK=8
export WANDER_MAX_WALK_LEN=128
export SKIP_EXISTING="${SKIP_EXISTING:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "=== NC walk grid ==="
echo "Repo: ${REPO_ROOT}"
echo "CKPT: ${CKPT}"
echo "Results: ${GRID_ROOT}"
echo "Datasets: ${DATASETS[*]}"

MANIFEST="${GRID_ROOT}/grid_manifest.tsv"
printf 'dataset\tn\tn_train\ttier\teval_walk_num\twalk_len\ttrain_kv_walk_num\trun_name\tmem\ttime\n' > "${MANIFEST}"

N_SUBMIT=0
N_SKIP=0

submit_cell() {
  local ds="$1" ew="$2" wl="$3" kv="$4" n="$5" ntrain="$6" tier="$7"
  local mem time_lim run_name jname metrics
  mem="$(dataset_mem "${ds}")"
  time_lim="$(dataset_time "${ds}")"
  run_name="ew${ew}_wl${wl}_kv${kv}"
  jname="ncg-${ds}-e${ew}-l${wl}-k${kv}"
  metrics="${GRID_ROOT}/${ds}/${run_name}/metrics.json"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${ds}" "${n}" "${ntrain}" "${tier}" "${ew}" "${wl}" "${kv}" "${run_name}" "${mem}" "${time_lim}" \
    >> "${MANIFEST}"

  if [[ "${SKIP_EXISTING}" == "1" && -s "${metrics}" ]]; then
    echo "  skip ${ds} ${run_name} (exists)"
    N_SKIP=$((N_SKIP + 1))
    return 0
  fi

  export TEST_DATASET="${ds}"
  export EVAL_RUN_DIR="${GRID_ROOT}/${ds}"
  export RUN_NAME="${run_name}"
  export WANDER_WALK_NUM="${ew}"
  export WANDER_EVAL_WALK_NUM="${ew}"
  export WANDER_WALK_LEN="${wl}"
  export WANDER_TRAIN_KV_WALK_NUM="${kv}"

  local -a cmd=(
    env
    "SLURM_NUM_GPUS=${SLURM_NUM_GPUS:-1}"
    "SLURM_JOB_NAME=${jname}"
    "CLUSTER_MEM=${mem}"
    "CLUSTER_TIME_EVAL=${time_lim}"
    bash "${SCRIPT_DIR}/submit.sh" "${JOB_SCRIPT}" "${ds}"
  )
  printf '+ %s %s\n' "${jname}" "${mem}"
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${cmd[@]}"
  fi
  N_SUBMIT=$((N_SUBMIT + 1))
}

GRID_TSV="$("${PYTHON}" - "${RAW_DATA_DIR}" "${DATASETS[@]}" <<'PY'
import sys
from data.dataset import DataSet, get_datasetargs

data_dir = sys.argv[1]
for name in sys.argv[2:]:
    args = get_datasetargs(name)
    ds = DataSet(args, pca_target_dim=9999)
    bundle = ds.load(data_dir=data_dir, seed=0)
    data = bundle.train
    n = int(data.num_nodes)
    n_train = int(data.train_mask.sum())
    if n < 1000:
        tier = "n<1k"
        ews = [32, 64, 128]
        kvs = [32, 64, 128]
    elif n_train < 1000:
        tier = "n_train<1k"
        ews = [64, 128]
        kvs = [64, 128]
    else:
        tier = "n_train>=1k"
        ews = [64, 128]
        kvs = [128, 256, 512]
    ews_s = ",".join(str(x) for x in ews)
    kvs_s = ",".join(str(x) for x in kvs)
    print(f"{name}\t{n}\t{n_train}\t{tier}\t{ews_s}\t{kvs_s}")
PY
)"

while IFS=$'\t' read -r ds n ntrain tier ews_s kvs_s; do
  echo "--- ${ds} n=${n} n_train=${ntrain} ${tier} ---"
  IFS=',' read -r -a ews <<< "${ews_s}"
  IFS=',' read -r -a kvs <<< "${kvs_s}"
  for ew in "${ews[@]}"; do
    for wl in 32 64 128; do
      for kv in "${kvs[@]}"; do
        submit_cell "${ds}" "${ew}" "${wl}" "${kv}" "${n}" "${ntrain}" "${tier}"
      done
    done
  done
done <<< "${GRID_TSV}"

echo "=== Done queueing (dry_run=${DRY_RUN} submitted=${N_SUBMIT} skipped=${N_SKIP}) ==="
echo "Manifest: ${MANIFEST}"
