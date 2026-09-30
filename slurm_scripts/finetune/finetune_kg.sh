#!/usr/bin/env bash
#SBATCH --job-name=wander-ft-kg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm_finetune_%x_%j.out
#SBATCH --error=logs/slurm_finetune_%x_%j.err

# Finetune checkpoints/pretrained_wander/ on one KG LP dataset.
#
# Paper protocol: lr=5e-5, weight_decay=0, 256 negatives, no-replacement deck.
# Walks n from kg_protocol.sh (default KG eval / Appendix). Walk length 128.
# One full pass over the training split per epoch, up to 5 epochs.
# Inline val (cheap ensembles) selects the highest-MRR checkpoint.
# --skip_final_test: zero-shot test is queue_eval_ft_kg.sh (same n/P as ZS).
#   wander_test_samples=2 for cheap val / checkpoint selection
#   wander_global_kv_neighbors=200
#
# Usage (from the repo root):
#   bash slurm_scripts/submit.sh slurm_scripts/finetune/finetune_kg.sh FB15K_237
#   bash slurm_scripts/finetune/queue_finetune_kg.sh

set -euo pipefail

export PYTHONUNBUFFERED=1

DATASET="${1:-${DATASET:-}}"
if [[ -z "${DATASET}" ]]; then
  echo "ERROR: pass a KG dataset name as first argument (e.g. FB15K_237)" >&2
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

_PROTO="${SLURM_SCRIPT_DIR:-}/kg_protocol.sh"
if [[ ! -f "${_PROTO}" ]]; then
  _PROTO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/kg_protocol.sh"
fi
# shellcheck source=/dev/null
source "${_PROTO}"
kg_lookup_protocol "${DATASET}"

if [[ -z "${INIT_CHECKPOINT:-}" ]]; then
  INIT_CHECKPOINT="$(cluster_default_pretrained_ckpt)"
fi
RUN_NAME="${RUN_NAME:-ft_${DATASET}_kg}"
LR="${LR:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
TRAIN_WITHOUT_REPLACEMENT="${TRAIN_WITHOUT_REPLACEMENT:-1}"
SAVE_CHECKPOINT_INTERVAL="${SAVE_CHECKPOINT_INTERVAL:-1}"
NUM_GRAPHS_BATCH="${NUM_GRAPHS_BATCH:-1}"
BATCH_SIZE_LINK="${BATCH_SIZE_LINK:-${KG_BATCH_SIZE}}"
EVAL_BATCH_SIZE_LINK="${EVAL_BATCH_SIZE_LINK:-8}"
WANDER_WALK_NUM="${WANDER_WALK_NUM:-${KG_WALK_NUM}}"
WANDER_WALK_LEN="${WANDER_WALK_LEN:-128}"
WANDER_MAX_WALK_LEN="${WANDER_MAX_WALK_LEN:-128}"
WANDER_ADAPTIVE_WALKS="${WANDER_ADAPTIVE_WALKS:-0}"
WANDER_TRAIN_KV_CHUNK_SIZE="${WANDER_TRAIN_KV_CHUNK_SIZE:-20000}"
WANDER_TRAIN_KV_PASSES="${WANDER_TRAIN_KV_PASSES:-16}"
WANDER_TEST_SAMPLES="${WANDER_TEST_SAMPLES:-2}"
WANDER_GLOBAL_KV_NEIGHBORS="${WANDER_GLOBAL_KV_NEIGHBORS:-200}"
CHECKPOINT_METRIC="${CHECKPOINT_METRIC:-mrr}"
MAX_EVAL_SAMPLES_TRAINING="${MAX_EVAL_SAMPLES_TRAINING:-2000}"
SKIP_FINAL_TEST="${SKIP_FINAL_TEST:-1}"

cluster_stage_datasets "${DATASET}"

_count_full_bpe() {
  "$PYTHON" - "${DATASET}" "${RAW_DATA_DIR}" "${BATCH_SIZE_LINK}" <<'PY'
import sys
from data.dataset import get_datasetargs

name, data_dir, bs = sys.argv[1], sys.argv[2], int(sys.argv[3])
info = get_datasetargs(name)
fn = info.get("load_graph_bundle")
if fn is None:
    raise SystemExit(f"{name} has no load_graph_bundle")
kwargs = {"add_inverse_edges_kgs": True}
ver = info.get("dataset_version")
if ver:
    kwargs["dataset_version"] = ver
bundle = fn(data_dir, **kwargs)
n_train = int(bundle.train.train_mask.sum().item())
print(max(1, (n_train + bs - 1) // bs))
PY
}

BATCH_PER_EPOCH="${BATCH_PER_EPOCH:-$(_count_full_bpe)}"
MAX_EPOCHS="${MAX_EPOCHS:-5}"
PATIENCE="${PATIENCE:-0}"
FINAL_TEST_WANDER_SAMPLES="${FINAL_TEST_WANDER_SAMPLES:-${KG_ENSEMBLE_P}}"

CKPT_ARGS=()
if cluster_maybe_auto_resume "${RUN_NAME}" "finetune KG ${DATASET}"; then
  CKPT_ARGS+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
else
  if [[ ! -f "${INIT_CHECKPOINT}" ]]; then
    echo "ERROR: checkpoint not found: ${INIT_CHECKPOINT}" >&2
    echo "Place weights at checkpoints/pretrained_wander/pretrained_wander.pt" >&2
    exit 1
  fi
  INIT_CHECKPOINT="$(cluster_resolve_path "${INIT_CHECKPOINT}")"
  CKPT_ARGS+=(--init_checkpoint "${INIT_CHECKPOINT}")
  echo "=== Finetune KG ${DATASET} from ${INIT_CHECKPOINT} (run: ${RUN_NAME}) ==="
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

EXTRA_ARGS=()
if [[ "${NO_EVALUATE_HEAD_PREDICTIONS:-0}" == "1" || "${NO_EVALUATE_HEAD_PREDICTIONS:-0}" == "true" ]]; then
  EXTRA_ARGS+=(--no-evaluate_head_predictions)
fi
if [[ "${TRAIN_WITHOUT_REPLACEMENT}" == "1" || "${TRAIN_WITHOUT_REPLACEMENT}" == "true" ]]; then
  EXTRA_ARGS+=(--train_without_replacement)
else
  EXTRA_ARGS+=(--no-train_without_replacement)
fi
if [[ "${SKIP_FINAL_TEST}" == "1" || "${SKIP_FINAL_TEST}" == "true" ]]; then
  EXTRA_ARGS+=(--skip_final_test)
fi

echo "=== KG FT: dataset=${DATASET} epochs=${MAX_EPOCHS} bpe=${BATCH_PER_EPOCH} (full train) patience=${PATIENCE} bs_link=${BATCH_SIZE_LINK} wd=${WEIGHT_DECAY} no_replace=${TRAIN_WITHOUT_REPLACEMENT} walks=${WANDER_WALK_NUM}x${WANDER_WALK_LEN} ts_val=${WANDER_TEST_SAMPLES} skip_final_test=${SKIP_FINAL_TEST} kv_nbrs=${WANDER_GLOBAL_KV_NEIGHBORS} ==="

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
  --patience "${PATIENCE}" \
  --num_graphs_batch "${NUM_GRAPHS_BATCH}" \
  --batch_size_link "${BATCH_SIZE_LINK}" \
  --eval_batch_size_link "${EVAL_BATCH_SIZE_LINK}" \
  --checkpoint_metric "${CHECKPOINT_METRIC}" \
  --num_negatives 256 \
  --save_checkpoint_interval "${SAVE_CHECKPOINT_INTERVAL}" \
  --max_eval_samples_training "${MAX_EVAL_SAMPLES_TRAINING}" \
  --wander_cached_train_kv \
  --wander_train_kv_passes "${WANDER_TRAIN_KV_PASSES}" \
  --wander_train_kv_chunk_size "${WANDER_TRAIN_KV_CHUNK_SIZE}" \
  --wander_test_samples "${WANDER_TEST_SAMPLES}" \
  --final_test_wander_samples "${FINAL_TEST_WANDER_SAMPLES}" \
  "${WALK_FLAGS[@]}" \
  --wander_inter_node_chunksize 128 \
  --wander_intra_node_chunksize 2048 \
  --wander_global_attention_update \
  --wander_global_kv_neighbors "${WANDER_GLOBAL_KV_NEIGHBORS}" \
  --node_cls_random_training_batches \
  --add_inverse_edges_KGs \
  --pca_target_dim 64 \
  --pca_target_dim_link 32 \
  --pca_target_dim_link_syn 32 \
  --no-pca_before_normalization \
  --final_inductive_zscore \
  --drop_constant_train_features \
  "${EXTRA_ARGS[@]}"

finish_job
