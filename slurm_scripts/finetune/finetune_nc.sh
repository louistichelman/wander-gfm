#!/usr/bin/env bash
#SBATCH --job-name=wander-ft-nc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm_finetune_%x_%j.out
#SBATCH --error=logs/slurm_finetune_%x_%j.err

# Finetune checkpoints/pretrained_wander/ on one NC dataset (features on).
#
# Protocol:
#   first split only (--nc_split_index 0)
#   batch_size=16, batch_per_epoch=5 (optimizer steps / epoch)
#   --nc_train_query_frac 0.25 --nc_train_query_cap 1024
#   max_epochs=100, patience=15, weight_decay=0
#   val cap: --max_eval_samples_training 2000
#   num_walks=128, walk_len by Table 8 pretrain adaptive schedule (n<1k -> 32, 1k-8k -> 64, n>=8k -> 128)
#   eval: cached train KV (3 passes val), proximity batches, eval batch 128
#   wander_test_samples=3 during training; --skip_final_test (grid has no test)
#   After the LR grid: queue_eval_ft_nc.sh (zero-shot walks + seeds, split 0)
#   LR grid is submitted by queue_finetune_nc.sh
#
# Usage (from the repo root):
#   bash slurm_scripts/submit.sh slurm_scripts/finetune/finetune_nc.sh CORA
#   LR=1.08e-4 RUN_NAME=ft_CORA_nc_lr1p08e-4 bash slurm_scripts/submit.sh slurm_scripts/finetune/finetune_nc.sh CORA
#
# Env overrides: INIT_CHECKPOINT RUN_NAME MAX_EPOCHS PATIENCE LR WEIGHT_DECAY
#   BATCH_SIZE_NODE EVAL_BATCH_SIZE_NODE BATCH_PER_EPOCH
#   NC_TRAIN_QUERY_FRAC NC_TRAIN_QUERY_CAP MAX_EVAL_SAMPLES_TRAINING
#   WANDER_TRAIN_KV_PASSES FINAL_TEST_TRAIN_KV_PASSES WANDER_TRAIN_KV_WALK_NUM
#   WANDER_WALK_NUM WANDER_EVAL_WALK_NUM WANDER_WALK_LEN FORCE_FRESH=1 RESUME_CHECKPOINT=...

set -euo pipefail

export PYTHONUNBUFFERED=1

DATASET="${1:-${DATASET:-}}"
if [[ -z "${DATASET}" ]]; then
  echo "ERROR: pass dataset name as first argument (e.g. CORA, CITESEER, CO_PHYSICS)" >&2
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

if [[ -z "${INIT_CHECKPOINT:-}" ]]; then
  INIT_CHECKPOINT="$(cluster_default_pretrained_ckpt)"
fi
RUN_NAME="${RUN_NAME:-ft_${DATASET}_nc}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
PATIENCE="${PATIENCE:-15}"
LR="${LR:-6.46e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
SAVE_CHECKPOINT_INTERVAL="${SAVE_CHECKPOINT_INTERVAL:-1}"
NUM_GRAPHS_BATCH="${NUM_GRAPHS_BATCH:-1}"
BATCH_SIZE_NODE="${BATCH_SIZE_NODE:-16}"
EVAL_BATCH_SIZE_NODE="${EVAL_BATCH_SIZE_NODE:-128}"
BATCH_PER_EPOCH="${BATCH_PER_EPOCH:-5}"
NC_TRAIN_QUERY_FRAC="${NC_TRAIN_QUERY_FRAC:-0.25}"
NC_TRAIN_QUERY_CAP="${NC_TRAIN_QUERY_CAP:-1024}"
WANDER_TRAIN_KV_CHUNK_SIZE="${WANDER_TRAIN_KV_CHUNK_SIZE:-20000}"
WANDER_TRAIN_KV_PASSES="${WANDER_TRAIN_KV_PASSES:-3}"
FINAL_TEST_TRAIN_KV_PASSES="${FINAL_TEST_TRAIN_KV_PASSES:-16}"
WANDER_TEST_SAMPLES="${WANDER_TEST_SAMPLES:-3}"
FINAL_TEST_WANDER_SAMPLES="${FINAL_TEST_WANDER_SAMPLES:-16}"
NC_SPLIT_INDEX="${NC_SPLIT_INDEX:-0}"
SKIP_FINAL_TEST="${SKIP_FINAL_TEST:-1}"
CHECKPOINT_METRIC="${CHECKPOINT_METRIC:-accuracy}"
MAX_EVAL_SAMPLES_TRAINING="${MAX_EVAL_SAMPLES_TRAINING:-2000}"
WANDER_WALK_NUM="${WANDER_WALK_NUM:-128}"
WANDER_EVAL_WALK_NUM="${WANDER_EVAL_WALK_NUM:-}"
WANDER_TRAIN_KV_WALK_NUM="${WANDER_TRAIN_KV_WALK_NUM:-}"
WANDER_ADAPTIVE_WALKS="${WANDER_ADAPTIVE_WALKS:-0}"

cluster_stage_datasets "${DATASET}"

read -r _n_train _n_nodes < <(
  "$PYTHON" - "${DATASET}" "${RAW_DATA_DIR}" <<'PY'
import sys
from data.dataset import DataSet, get_datasetargs

name, data_dir = sys.argv[1], sys.argv[2]
ds = DataSet(get_datasetargs(name))
data = ds._load_base_dataset_fn(data_dir, **ds._graphland_load_kwargs())
n_nodes = int(data.num_nodes)
splits = ds._create_node_level_splits(data, seed=0)
if splits is None:
    raise SystemExit(f"{name}: load_base_dataset produced no node split")
train_mask = splits[0]
if train_mask.ndim > 1:
    train_mask = train_mask[:, 0]
print(int(train_mask.sum().item()), n_nodes)
PY
)

if [[ -z "${WANDER_WALK_LEN:-}" ]]; then
  if ((_n_nodes >= 8000)); then
    WANDER_WALK_LEN=128
  elif ((_n_nodes >= 1000)); then
    WANDER_WALK_LEN=64
  else
    WANDER_WALK_LEN=32
  fi
fi
WANDER_MAX_WALK_LEN="${WANDER_MAX_WALK_LEN:-128}"

CKPT_ARGS=()
if cluster_maybe_auto_resume "${RUN_NAME}" "finetune NC ${DATASET}"; then
  CKPT_ARGS+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
else
  if [[ ! -f "${INIT_CHECKPOINT}" ]]; then
    echo "ERROR: checkpoint not found: ${INIT_CHECKPOINT}" >&2
    echo "Place weights at checkpoints/pretrained_wander/pretrained_wander.pt" >&2
    exit 1
  fi
  INIT_CHECKPOINT="$(cluster_resolve_path "${INIT_CHECKPOINT}")"
  CKPT_ARGS+=(--init_checkpoint "${INIT_CHECKPOINT}")
  echo "=== Finetune NC ${DATASET} from ${INIT_CHECKPOINT} (run: ${RUN_NAME}) ==="
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
if [[ -n "${WANDER_EVAL_WALK_NUM}" ]]; then
  EXTRA_ARGS+=(--wander_eval_walk_num "${WANDER_EVAL_WALK_NUM}")
fi
if [[ -n "${WANDER_TRAIN_KV_WALK_NUM}" ]]; then
  EXTRA_ARGS+=(--wander_train_kv_walk_num "${WANDER_TRAIN_KV_WALK_NUM}")
fi
if [[ "${DATASET}" == "CYCLES" || "${DATASET}" == "GRIDS" ]]; then
  IGNORE_FEATURES=1
fi
if [[ "${IGNORE_FEATURES:-0}" == "1" || "${IGNORE_FEATURES:-0}" == "true" ]]; then
  EXTRA_ARGS+=(--ignore_features)
fi
if [[ -n "${NC_SPLIT_INDEX}" ]]; then
  EXTRA_ARGS+=(--nc_split_index "${NC_SPLIT_INDEX}")
fi
if [[ "${SKIP_FINAL_TEST}" == "1" || "${SKIP_FINAL_TEST}" == "true" ]]; then
  EXTRA_ARGS+=(--skip_final_test)
fi

echo "=== NC FT: dataset=${DATASET} split=${NC_SPLIT_INDEX:-default} n=${_n_nodes} train_nodes=${_n_train} lr=${LR} wd=${WEIGHT_DECAY} epochs=${MAX_EPOCHS} patience=${PATIENCE} bpe=${BATCH_PER_EPOCH} batch_node=${BATCH_SIZE_NODE} eval_batch_node=${EVAL_BATCH_SIZE_NODE} eval_cap=${MAX_EVAL_SAMPLES_TRAINING} query_frac=${NC_TRAIN_QUERY_FRAC} query_cap=${NC_TRAIN_QUERY_CAP} walks=${WANDER_WALK_NUM}x${WANDER_WALK_LEN} eval_walks=${WANDER_EVAL_WALK_NUM:-${WANDER_WALK_NUM}} kv_walks=${WANDER_TRAIN_KV_WALK_NUM:-${WANDER_WALK_NUM}} max_walk_len=${WANDER_MAX_WALK_LEN} adaptive=${WANDER_ADAPTIVE_WALKS} kv_chunk=${WANDER_TRAIN_KV_CHUNK_SIZE} kv_passes_train=${WANDER_TRAIN_KV_PASSES} kv_passes_final=${FINAL_TEST_TRAIN_KV_PASSES} ts_train=${WANDER_TEST_SAMPLES} skip_final_test=${SKIP_FINAL_TEST} no_replace=${TRAIN_WITHOUT_REPLACEMENT} ==="

run_main \
  "${CKPT_ARGS[@]}" \
  --run_name "${RUN_NAME}" \
  --data_dir "${RAW_DATA_DIR}" \
  --save_load_path "${CHECKPOINTS_DIR}" \
  --train_datasets "${DATASET}" \
  --test_datasets "${DATASET}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --patience "${PATIENCE}" \
  --batch_per_epoch "${BATCH_PER_EPOCH}" \
  --max_epochs "${MAX_EPOCHS}" \
  --num_graphs_batch "${NUM_GRAPHS_BATCH}" \
  --batch_size_node "${BATCH_SIZE_NODE}" \
  --eval_batch_size_node "${EVAL_BATCH_SIZE_NODE}" \
  --nc_train_query_frac "${NC_TRAIN_QUERY_FRAC}" \
  --nc_train_query_cap "${NC_TRAIN_QUERY_CAP}" \
  --batch_size_link 4 \
  --eval_batch_size_link 8 \
  --checkpoint_metric "${CHECKPOINT_METRIC}" \
  --num_negatives 512 \
  --save_checkpoint_interval "${SAVE_CHECKPOINT_INTERVAL}" \
  --max_eval_samples_training "${MAX_EVAL_SAMPLES_TRAINING}" \
  --wander_cached_train_kv \
  --wander_train_kv_passes "${WANDER_TRAIN_KV_PASSES}" \
  --final_test_train_kv_passes "${FINAL_TEST_TRAIN_KV_PASSES}" \
  --wander_train_kv_chunk_size "${WANDER_TRAIN_KV_CHUNK_SIZE}" \
  --wander_test_samples "${WANDER_TEST_SAMPLES}" \
  --final_test_wander_samples "${FINAL_TEST_WANDER_SAMPLES}" \
  --nc_proximity_batching \
  "${WALK_FLAGS[@]}" \
  --wander_inter_node_chunksize 128 \
  --wander_intra_node_chunksize 2048 \
  --wander_refinements 6 \
  --wander_checkpoint_refinements \
  --wander_rotary_emb_intra_node_attention \
  --wander_global_attention_update \
  --wander_feature_embedding_mlp \
  --wander_randomize_feat_columns \
  --wander_randomize_label_columns \
  --wander_net gru \
  --node_cls_random_training_batches \
  --add_inverse_edges_KGs \
  --pca_target_dim 64 \
  --pca_target_dim_link 32 \
  --pca_target_dim_link_syn 32 \
  --pca_target_dim_node_syn 64 \
  --no-pca_before_normalization \
  --adaptive_pca_node_threshold 8000 \
  --final_inductive_zscore \
  --graphland_categorical_as_ordinals \
  --graphland_different_transform \
  --drop_constant_train_features \
  "${EXTRA_ARGS[@]}"

finish_job
