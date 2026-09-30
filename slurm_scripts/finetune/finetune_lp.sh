#!/usr/bin/env bash
#SBATCH --job-name=wander-ft-lp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm_finetune_%x_%j.out
#SBATCH --error=logs/slurm_finetune_%x_%j.err

# Finetune checkpoints/pretrained_wander/ on one ordinary (AnyGraph) LP dataset.
#
# Protocol:
#   lr=5e-5, weight_decay=0, num_negatives=256
#   bs_link=4 featured / 8 no-feature; max_epochs=5; no-replacement deck
#   batch_per_epoch = min(5000, ceil(n_train / bs_link))  (full pass, capped)
#   --skip_eval by default (no inline val; save every epoch)
#   Set SKIP_EVAL=0 for per-epoch eval on a capped test subset
#     (MAX_EVAL_SAMPLES_TRAINING, WANDER_TEST_SAMPLES).
#   PCA matches zero-shot LP: featured 64/64, no-feature 64/32
#   Featured graphs: 64x64 walks. No-feature: 128x128. Adaptive off.
#   After FT, evaluate completed epoch 5 (file epoch4) with the same protocol:
#     bash slurm_scripts/eval/queue_eval_ft_lp.sh
#
# Usage (from the repo root):
#   bash slurm_scripts/submit.sh slurm_scripts/finetune/finetune_lp.sh ANYGRAPH_CORA
#   bash slurm_scripts/finetune/queue_finetune_lp.sh

set -euo pipefail

export PYTHONUNBUFFERED=1

DATASET="${1:-${DATASET:-}}"
if [[ -z "${DATASET}" ]]; then
  echo "ERROR: pass an LP dataset name as first argument (e.g. ANYGRAPH_CORA)" >&2
  exit 1
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
source "${_slurm}/source_common.sh"

dataset_has_features() {
  local ds="$1"
  case "${ds}" in
    ANYGRAPH_CITESEER|ANYGRAPH_CORA|ANYGRAPH_PUBMED|ANYGRAPH_CS|ANYGRAPH_PRODUCTS_HOME)
      return 0 ;;
    *) return 1 ;;
  esac
}

if [[ -z "${INIT_CHECKPOINT:-}" ]]; then
  INIT_CHECKPOINT="$(cluster_default_pretrained_ckpt)"
fi
RUN_NAME="${RUN_NAME:-ft_${DATASET}_lp}"
MAX_EPOCHS="${MAX_EPOCHS:-5}"
LR="${LR:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
SAVE_CHECKPOINT_INTERVAL="${SAVE_CHECKPOINT_INTERVAL:-1}"
NUM_GRAPHS_BATCH="${NUM_GRAPHS_BATCH:-1}"
if dataset_has_features "${DATASET}"; then
  BATCH_SIZE_LINK="${BATCH_SIZE_LINK:-4}"
  WANDER_WALK_NUM="${WANDER_WALK_NUM:-64}"
  WANDER_WALK_LEN="${WANDER_WALK_LEN:-64}"
else
  BATCH_SIZE_LINK="${BATCH_SIZE_LINK:-8}"
  WANDER_WALK_NUM="${WANDER_WALK_NUM:-128}"
  WANDER_WALK_LEN="${WANDER_WALK_LEN:-128}"
fi
EVAL_BATCH_SIZE_LINK="${EVAL_BATCH_SIZE_LINK:-8}"
BPE_CAP="${BPE_CAP:-5000}"
WANDER_MAX_WALK_LEN="${WANDER_MAX_WALK_LEN:-128}"
WANDER_ADAPTIVE_WALKS="${WANDER_ADAPTIVE_WALKS:-0}"
WANDER_TRAIN_KV_CHUNK_SIZE="${WANDER_TRAIN_KV_CHUNK_SIZE:-20000}"
WANDER_TRAIN_KV_PASSES="${WANDER_TRAIN_KV_PASSES:-16}"
PCA_TARGET_DIM="${PCA_TARGET_DIM:-64}"
if dataset_has_features "${DATASET}"; then
  PCA_TARGET_DIM_LINK="${PCA_TARGET_DIM_LINK:-64}"
else
  PCA_TARGET_DIM_LINK="${PCA_TARGET_DIM_LINK:-32}"
fi
SKIP_EVAL="${SKIP_EVAL:-1}"
MAX_EVAL_SAMPLES_TRAINING="${MAX_EVAL_SAMPLES_TRAINING:-200}"
WANDER_TEST_SAMPLES="${WANDER_TEST_SAMPLES:-3}"
FINAL_TEST_WANDER_SAMPLES="${FINAL_TEST_WANDER_SAMPLES:-16}"
FINAL_TEST_TRAIN_KV_PASSES="${FINAL_TEST_TRAIN_KV_PASSES:-16}"
PATIENCE="${PATIENCE:-0}"

cluster_stage_datasets "${DATASET}"

_count_lp_train_queries() {
  "$PYTHON" - "${DATASET}" "${RAW_DATA_DIR}" <<'PY'
import sys
from data.dataset import get_datasetargs

name, data_dir = sys.argv[1], sys.argv[2]
info = get_datasetargs(name)
fn = info.get("load_graph_bundle")
if fn is None:
    raise SystemExit(f"{name} has no load_graph_bundle")
kwargs = {"add_inverse_edges_kgs": True}
ver = info.get("dataset_version")
if ver:
    kwargs["dataset_version"] = ver
bundle = fn(data_dir, **kwargs)
print(int(bundle.train.train_mask.sum().item()))
PY
}

if [[ -z "${BATCH_PER_EPOCH:-}" ]]; then
  _n_train="$(_count_lp_train_queries)"
  _full_bpe=$(( (_n_train + BATCH_SIZE_LINK - 1) / BATCH_SIZE_LINK ))
  if (( _full_bpe < 1 )); then
    _full_bpe=1
  fi
  if (( _full_bpe > BPE_CAP )); then
    BATCH_PER_EPOCH="${BPE_CAP}"
  else
    BATCH_PER_EPOCH="${_full_bpe}"
  fi
  echo "=== LP BPE: n_train=${_n_train} bs=${BATCH_SIZE_LINK} full=${_full_bpe} cap=${BPE_CAP} -> bpe=${BATCH_PER_EPOCH} ==="
fi

CKPT_ARGS=()
if cluster_maybe_auto_resume "${RUN_NAME}" "finetune LP ${DATASET}"; then
  CKPT_ARGS+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
else
  if [[ ! -f "${INIT_CHECKPOINT}" ]]; then
    echo "ERROR: checkpoint not found: ${INIT_CHECKPOINT}" >&2
    echo "Place weights at checkpoints/pretrained_wander/pretrained_wander.pt" >&2
    exit 1
  fi
  INIT_CHECKPOINT="$(cluster_resolve_path "${INIT_CHECKPOINT}")"
  CKPT_ARGS+=(--init_checkpoint "${INIT_CHECKPOINT}")
  echo "=== Finetune LP ${DATASET} from ${INIT_CHECKPOINT} (run: ${RUN_NAME}) ==="
fi

WALK_FLAGS=(
  --wander_walk_num "${WANDER_WALK_NUM}"
  --wander_walk_len "${WANDER_WALK_LEN}"
  --wander_max_walk_len "${WANDER_MAX_WALK_LEN}"
)
if [[ "${WANDER_ADAPTIVE_WALKS}" == "1" || "${WANDER_ADAPTIVE_WALKS}" == "true" ]]; then
  WALK_FLAGS+=(--wander_adaptive_walks)
else
  WALK_FLAGS+=(--no_wander_adaptive_walks)
fi

TRAIN_WITHOUT_REPLACEMENT="${TRAIN_WITHOUT_REPLACEMENT:-1}"
EXTRA_ARGS=()
if [[ "${TRAIN_WITHOUT_REPLACEMENT}" == "1" || "${TRAIN_WITHOUT_REPLACEMENT}" == "true" ]]; then
  EXTRA_ARGS+=(--train_without_replacement)
else
  EXTRA_ARGS+=(--no-train_without_replacement)
fi
if [[ "${SKIP_EVAL}" == "1" || "${SKIP_EVAL}" == "true" ]]; then
  EXTRA_ARGS+=(--skip_eval)
else
  EXTRA_ARGS+=(
    --max_eval_samples_training "${MAX_EVAL_SAMPLES_TRAINING}"
    --wander_test_samples "${WANDER_TEST_SAMPLES}"
    --final_test_wander_samples "${FINAL_TEST_WANDER_SAMPLES}"
    --final_test_train_kv_passes "${FINAL_TEST_TRAIN_KV_PASSES}"
    --patience "${PATIENCE}"
  )
fi

echo "=== LP FT: dataset=${DATASET} epochs=${MAX_EPOCHS} bpe=${BATCH_PER_EPOCH} bs=${BATCH_SIZE_LINK} lr=${LR} wd=${WEIGHT_DECAY} negs=256 walks=${WANDER_WALK_NUM}x${WANDER_WALK_LEN} pca=${PCA_TARGET_DIM}/${PCA_TARGET_DIM_LINK} skip_eval=${SKIP_EVAL} eval_cap=${MAX_EVAL_SAMPLES_TRAINING} ts=${WANDER_TEST_SAMPLES}/${FINAL_TEST_WANDER_SAMPLES} no_replace=${TRAIN_WITHOUT_REPLACEMENT} ==="

run_main \
  "${CKPT_ARGS[@]}" \
  --run_name "${RUN_NAME}" \
  --data_dir "${RAW_DATA_DIR}" \
  --save_load_path "${CHECKPOINTS_DIR}" \
  --train_datasets "${DATASET}" \
  --test_datasets "${DATASET}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --batch_per_epoch "${BATCH_PER_EPOCH}" \
  --max_epochs "${MAX_EPOCHS}" \
  --num_graphs_batch "${NUM_GRAPHS_BATCH}" \
  --batch_size_link "${BATCH_SIZE_LINK}" \
  --eval_batch_size_link "${EVAL_BATCH_SIZE_LINK}" \
  --num_negatives 256 \
  --save_checkpoint_interval "${SAVE_CHECKPOINT_INTERVAL}" \
  --wander_cached_train_kv \
  --wander_train_kv_passes "${WANDER_TRAIN_KV_PASSES}" \
  --wander_train_kv_chunk_size "${WANDER_TRAIN_KV_CHUNK_SIZE}" \
  "${WALK_FLAGS[@]}" \
  --wander_inter_node_chunksize 128 \
  --wander_intra_node_chunksize 2048 \
  --wander_global_attention_update \
  --node_cls_random_training_batches \
  --add_inverse_edges_KGs \
  --pca_target_dim "${PCA_TARGET_DIM}" \
  --pca_target_dim_link "${PCA_TARGET_DIM_LINK}" \
  --pca_target_dim_link_syn 32 \
  --no-pca_before_normalization \
  --final_inductive_zscore \
  --drop_constant_train_features \
  "${EXTRA_ARGS[@]}"

finish_job
