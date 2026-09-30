#!/usr/bin/env bash
#SBATCH --job-name=wander-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-24:00:00
#SBATCH --output=logs/slurm_eval_%x_%j.out
#SBATCH --error=logs/slurm_eval_%x_%j.err

# Zero-shot eval of checkpoints/pretrained_wander/ on one dataset.
# Zero-shot eval protocol:
#   cached train KV, proximity NC, eval batch node 128 / link 8, kv_chunk=20000,
#   adaptive walks off, max_walk_len=128.
# NC size-tier walks (SIZE_TIER_WALKS=1; not the Table 8 train adaptive 1k/8k schedule):
#   n<=1k -> 32x32; 1k<n<=10k -> 64x128; 10k<n<=40k -> 128x128; n>40k -> 256x128.
# SIZE_TIER_WALK_NUM_DIV=2|4 scales K_base only (walk length unchanged).
# SIZE_TIER_WALK_LEN_DIV=2|4 scales walk length only (K_base unchanged).
# NC_SPLIT_INDEX: official mask column or GraphAny sklearn seed (queue_eval_nc.sh).
# KG: queue_eval_kg.sh sets per-dataset wander_walk_num (Flock n) and
#   wander_test_samples (Flock P); walk_len stays 128. Fast uniform and compile
#   default on (eval-only compile; RMSNorm stays eager).
# Ordinary LP: Recall@20.
# NC proximity: packed-ball batching, --nc_proximity_batch_max_radius 8
#   (override with NC_PROXIMITY_BATCH_MAX_RADIUS).
#
# Usage (from the repo root):
#   bash slurm_scripts/submit.sh slurm_scripts/eval/eval_one_dataset.sh CORA
#   SLURM_NUM_GPUS=2 bash slurm_scripts/eval/queue_eval_nc.sh
#     -> torchrun shards Phase B test batches across GPUs after train-KV.
#   Prefer queue_eval_{nc,kg,lp}.sh for the full suites.

set -euo pipefail

export PYTHONUNBUFFERED=1

TEST_DATASET="${1:-${TEST_DATASET:-}}"
if [[ -z "${TEST_DATASET}" ]]; then
  echo "ERROR: pass dataset name as first argument or set TEST_DATASET" >&2
  exit 1
fi
NC_SPLIT_INDEX="${NC_SPLIT_INDEX:-}"
RUN_NAME="${RUN_NAME:-${TEST_DATASET}}"

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

cluster_stage_datasets "${TEST_DATASET}"

if [[ -z "${CKPT:-}" ]]; then
  CKPT="$(cluster_default_pretrained_ckpt)"
fi
EVAL_RUN_DIR="${EVAL_RUN_DIR:-${RESULTS_DIR}/eval}"
WANDER_TEST_SAMPLES="${WANDER_TEST_SAMPLES:-16}"
WANDER_TRAIN_KV_PASSES="${WANDER_TRAIN_KV_PASSES:-16}"
WANDER_TRAIN_KV_CHUNK_SIZE="${WANDER_TRAIN_KV_CHUNK_SIZE:-20000}"
EVAL_SEED="${EVAL_SEED:-0}"
# EVAL_SEEDS (colon/comma/space) overrides single EVAL_SEED.
# NC / ordinary LP queues default 0:1:2. UniLP queues use 0:1:2:3:4:5:6.
if [[ -n "${EVAL_SEEDS:-}" ]]; then
  EVAL_SEEDS_RAW="${EVAL_SEEDS//:/ }"
  EVAL_SEEDS_RAW="${EVAL_SEEDS_RAW//,/ }"
  # shellcheck disable=SC2206
  EVAL_SEEDS=(${EVAL_SEEDS_RAW})
else
  EVAL_SEEDS=("${EVAL_SEED}")
fi
BATCH_SIZE_LINK="${BATCH_SIZE_LINK:-4}"
EVAL_BATCH_SIZE_LINK="${EVAL_BATCH_SIZE_LINK:-8}"
EVAL_BATCH_SIZE_NODE="${EVAL_BATCH_SIZE_NODE:-128}"
if [[ "${TEST_DATASET}" == "GRIDS" || "${TEST_DATASET}" == "CYCLES" ]]; then
  IGNORE_FEATURES=1
else
  IGNORE_FEATURES="${IGNORE_FEATURES:-0}"
fi
IGNORE_EDGE_TYPES="${IGNORE_EDGE_TYPES:-0}"
WANDER_ADAPTIVE_WALKS="${WANDER_ADAPTIVE_WALKS:-0}"
WANDER_TRAIN_KV_WALK_NUM="${WANDER_TRAIN_KV_WALK_NUM:-}"
WANDER_WALK_NUM="${WANDER_WALK_NUM:-}"
WANDER_WALK_LEN="${WANDER_WALK_LEN:-}"
WANDER_MAX_WALK_LEN="${WANDER_MAX_WALK_LEN:-128}"
SIZE_TIER_WALKS="${SIZE_TIER_WALKS:-0}"
SIZE_TIER_WALK_NUM_DIV="${SIZE_TIER_WALK_NUM_DIV:-1}"
SIZE_TIER_WALK_LEN_DIV="${SIZE_TIER_WALK_LEN_DIV:-1}"
PCA_TARGET_DIM="${PCA_TARGET_DIM:-64}"
PCA_TARGET_DIM_LINK="${PCA_TARGET_DIM_LINK:-32}"
ADAPTIVE_PCA_NODE_THRESHOLD="${ADAPTIVE_PCA_NODE_THRESHOLD:-0}"

if [[ -n "${EVAL_RUN_DIR}" && "${EVAL_RUN_DIR}" == "${HOME_RESULTS_DIR}"/* ]]; then
  EVAL_RUN_DIR="${RESULTS_DIR}/${EVAL_RUN_DIR#${HOME_RESULTS_DIR}/}"
fi

CKPT="$(cluster_resolve_path "${CKPT}")"
if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT}" >&2
  echo "Place weights at checkpoints/pretrained_wander/pretrained_wander.pt (e.g. pretrained_wander.pt)" >&2
  exit 1
fi

SAVE_LOAD_PATH="${EVAL_RUN_DIR}"
mkdir -p "${SAVE_LOAD_PATH}"
METRICS_FILE="${SAVE_LOAD_PATH}/${RUN_NAME}/metrics.json"
if [[ "${SKIP_EXISTING:-0}" == "1" && -s "${METRICS_FILE}" ]]; then
  echo "=== Skip existing ${TEST_DATASET} ${RUN_NAME}: ${METRICS_FILE} ==="
  finish_job
  exit 0
fi

if [[ "${SIZE_TIER_WALKS}" == "1" || "${SIZE_TIER_WALKS}" == "true" ]]; then
  read -r _n_nodes < <(
    "$PYTHON" - "${TEST_DATASET}" "${RAW_DATA_DIR}" <<'PY'
import sys
from data.datasets import DATASET_REGISTRY

name, data_dir = sys.argv[1], sys.argv[2]
mod = DATASET_REGISTRY[name]
data = mod.load_base_dataset(data_dir)
print(int(data.num_nodes))
PY
  )
  if ((_n_nodes > 40000)); then
    _num=256
    _len=128
  elif ((_n_nodes > 10000)); then
    _num=128
    _len=128
  elif ((_n_nodes > 1000)); then
    _num=64
    _len=128
  else
    _num=32
    _len=32
  fi
  if [[ "${SIZE_TIER_WALK_NUM_DIV}" != "1" ]]; then
    _num=$((_num / SIZE_TIER_WALK_NUM_DIV))
    if ((_num < 1)); then
      _num=1
    fi
  fi
  if [[ "${SIZE_TIER_WALK_LEN_DIV}" != "1" ]]; then
    _len=$((_len / SIZE_TIER_WALK_LEN_DIV))
    if ((_len < 1)); then
      _len=1
    fi
  fi
  WANDER_WALK_LEN="${WANDER_WALK_LEN:-${_len}}"
  WANDER_WALK_NUM="${WANDER_WALK_NUM:-${_num}}"
else
  WANDER_WALK_NUM="${WANDER_WALK_NUM:-128}"
  WANDER_WALK_LEN="${WANDER_WALK_LEN:-128}"
fi

# Phase B: sample 2x walks then keep only those that visit a train node (p=0).
if [[ -n "${WANDER_KEEP_TRAIN_FREE_P:-}" && -z "${WANDER_EVAL_WALK_NUM:-}" ]]; then
  _filt_mult="${WANDER_FILTER_WALK_NUM_MULT:-2}"
  WANDER_EVAL_WALK_NUM="$((WANDER_WALK_NUM * _filt_mult))"
fi

EXTRA_ARGS=()
if [[ "${IGNORE_FEATURES}" == "1" || "${IGNORE_FEATURES}" == "true" ]]; then
  EXTRA_ARGS+=(--ignore_features)
fi
if [[ "${IGNORE_EDGE_TYPES}" == "1" || "${IGNORE_EDGE_TYPES}" == "true" ]]; then
  EXTRA_ARGS+=(--ignore_edge_types)
fi
if [[ "${WANDER_ADAPTIVE_WALKS}" == "1" || "${WANDER_ADAPTIVE_WALKS}" == "true" ]]; then
  EXTRA_ARGS+=(--wander_adaptive_walks)
else
  EXTRA_ARGS+=(--no_wander_adaptive_walks)
fi
if [[ "${WANDER_FAST_UNIFORM_WALKS:-1}" == "1" || "${WANDER_FAST_UNIFORM_WALKS:-1}" == "true" ]]; then
  EXTRA_ARGS+=(--wander_fast_uniform_walks)
else
  EXTRA_ARGS+=(--no_wander_fast_uniform_walks)
fi
if [[ "${WANDER_COMPILE_EVAL:-1}" == "1" || "${WANDER_COMPILE_EVAL:-1}" == "true" ]]; then
  EXTRA_ARGS+=(--wander_compile_eval)
else
  EXTRA_ARGS+=(--no_wander_compile_eval)
fi
if [[ -n "${WANDER_TRAIN_KV_WALK_NUM}" ]]; then
  EXTRA_ARGS+=(--wander_train_kv_walk_num "${WANDER_TRAIN_KV_WALK_NUM}")
fi
if [[ -n "${WANDER_EVAL_WALK_NUM:-}" ]]; then
  EXTRA_ARGS+=(--wander_eval_walk_num "${WANDER_EVAL_WALK_NUM}")
fi
if [[ -n "${WANDER_KEEP_TRAIN_FREE_P:-}" ]]; then
  EXTRA_ARGS+=(--wander_keep_train_free_p "${WANDER_KEEP_TRAIN_FREE_P}")
fi
if [[ -n "${LINK_PRED_EVAL:-}" ]]; then
  EXTRA_ARGS+=(--link_pred_eval "${LINK_PRED_EVAL}")
fi
if [[ -n "${SAMPLED_RECALL_NUM_NEG:-}" ]]; then
  EXTRA_ARGS+=(--sampled_recall_num_neg "${SAMPLED_RECALL_NUM_NEG}")
fi
if [[ -n "${SAMPLED_RECALL_MAX_SOURCES:-}" ]]; then
  EXTRA_ARGS+=(--sampled_recall_max_sources "${SAMPLED_RECALL_MAX_SOURCES}")
fi
if [[ "${ADAPTIVE_PCA_NODE_THRESHOLD}" != "0" ]]; then
  EXTRA_ARGS+=(--adaptive_pca_node_threshold "${ADAPTIVE_PCA_NODE_THRESHOLD}")
fi
if [[ "${NO_EVALUATE_HEAD_PREDICTIONS:-0}" == "1" || "${NO_EVALUATE_HEAD_PREDICTIONS:-0}" == "true" ]]; then
  EXTRA_ARGS+=(--no-evaluate_head_predictions)
fi
if [[ -n "${WANDER_GLOBAL_KV_NEIGHBORS:-}" ]]; then
  EXTRA_ARGS+=(--wander_global_kv_neighbors "${WANDER_GLOBAL_KV_NEIGHBORS}")
fi
if [[ -n "${NC_SPLIT_INDEX}" ]]; then
  EXTRA_ARGS+=(--nc_split_index "${NC_SPLIT_INDEX}")
fi
DROP_CONSTANT_TRAIN_FEATURES="${DROP_CONSTANT_TRAIN_FEATURES:-1}"
if [[ "${DROP_CONSTANT_TRAIN_FEATURES}" == "1" || "${DROP_CONSTANT_TRAIN_FEATURES}" == "true" ]]; then
  EXTRA_ARGS+=(--drop_constant_train_features)
else
  EXTRA_ARGS+=(--no_drop_constant_train_features)
fi
# Both of these derive column statistics from train nodes only.
FINAL_INDUCTIVE_ZSCORE="${FINAL_INDUCTIVE_ZSCORE:-1}"
if [[ "${FINAL_INDUCTIVE_ZSCORE}" == "1" || "${FINAL_INDUCTIVE_ZSCORE}" == "true" ]]; then
  EXTRA_ARGS+=(--final_inductive_zscore)
else
  EXTRA_ARGS+=(--no_final_inductive_zscore)
fi
NC_PROXIMITY_BATCH_MAX_RADIUS="${NC_PROXIMITY_BATCH_MAX_RADIUS:-8}"
if [[ -n "${NC_PROXIMITY_BATCH_MAX_RADIUS}" ]]; then
  EXTRA_ARGS+=(--nc_proximity_batch_max_radius "${NC_PROXIMITY_BATCH_MAX_RADIUS}")
fi
if [[ -n "${EXTRA_CLI:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS+=(${EXTRA_CLI})
fi
if [[ -n "${EVAL_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  _eval_extra=(${EVAL_EXTRA_ARGS})
  EXTRA_ARGS+=("${_eval_extra[@]}")
  echo "=== EVAL_EXTRA_ARGS: ${EVAL_EXTRA_ARGS} ==="
fi

echo "=== Evaluating ${TEST_DATASET} from ${CKPT} (run=${RUN_NAME}, split=${NC_SPLIT_INDEX:-default}, seeds=${EVAL_SEEDS[*]}, walks=${WANDER_WALK_NUM}x${WANDER_WALK_LEN}, eval_walks=${WANDER_EVAL_WALK_NUM:-}, keep_train_free_p=${WANDER_KEEP_TRAIN_FREE_P:-}, max_len=${WANDER_MAX_WALK_LEN}, ts=${WANDER_TEST_SAMPLES}, kv_passes=${WANDER_TRAIN_KV_PASSES}, link_bs=${BATCH_SIZE_LINK}, eval_link_bs=${EVAL_BATCH_SIZE_LINK}, kv_chunk=${WANDER_TRAIN_KV_CHUNK_SIZE}, adaptive=${WANDER_ADAPTIVE_WALKS}, size_tier=${SIZE_TIER_WALKS}, walk_num_div=${SIZE_TIER_WALK_NUM_DIV}, pca=${PCA_TARGET_DIM}, pca_link=${PCA_TARGET_DIM_LINK}, drop_constant=${DROP_CONSTANT_TRAIN_FEATURES}, final_zscore=${FINAL_INDUCTIVE_ZSCORE}, fast_uniform=${WANDER_FAST_UNIFORM_WALKS:-1}, compile_eval=${WANDER_COMPILE_EVAL:-1}, proximity_max_radius=${NC_PROXIMITY_BATCH_MAX_RADIUS}, extra_cli=${EXTRA_CLI:-}, gpus=${NPROC_PER_NODE:-1}) ==="
echo "=== Metrics directory: ${SAVE_LOAD_PATH}/${RUN_NAME} ==="
echo "=== Batch resume: ${SAVE_LOAD_PATH}/${RUN_NAME}/eval_batch_resume (resubmit with the same EVAL_RUN_DIR; FORCE_FRESH=1 to start over) ==="

run_main \
  --init_checkpoint "${CKPT}" \
  --eval_only \
  --data_dir "${RAW_DATA_DIR}" \
  --save_load_path "${SAVE_LOAD_PATH}" \
  --run_name "${RUN_NAME}" \
  --test_datasets "${TEST_DATASET}" \
  --batch_size_link "${BATCH_SIZE_LINK}" \
  --eval_batch_size_link "${EVAL_BATCH_SIZE_LINK}" \
  --batch_size_node 16 \
  --eval_batch_size_node "${EVAL_BATCH_SIZE_NODE}" \
  --wander_walk_num "${WANDER_WALK_NUM}" \
  --wander_walk_len "${WANDER_WALK_LEN}" \
  --wander_max_walk_len "${WANDER_MAX_WALK_LEN}" \
  --wander_global_attention_update \
  --wander_inter_node_chunksize 128 \
  --wander_intra_node_chunksize 2048 \
  --wander_refinements 6 \
  --wander_checkpoint_refinements \
  --wander_rotary_emb_intra_node_attention \
  --wander_feature_embedding_mlp \
  --wander_randomize_feat_columns \
  --wander_randomize_label_columns \
  --wander_net gru \
  --node_cls_random_training_batches \
  --add_inverse_edges_KGs \
  --num_negatives 512 \
  --wander_test_samples "${WANDER_TEST_SAMPLES}" \
  --wander_cached_train_kv \
  --nc_proximity_batching \
  --wander_train_kv_passes "${WANDER_TRAIN_KV_PASSES}" \
  --wander_train_kv_chunk_size "${WANDER_TRAIN_KV_CHUNK_SIZE}" \
  --pca_target_dim "${PCA_TARGET_DIM}" \
  --pca_target_dim_link "${PCA_TARGET_DIM_LINK}" \
  --pca_target_dim_link_syn 32 \
  --pca_target_dim_node_syn 64 \
  --no-pca_before_normalization \
  --graphland_categorical_as_ordinals \
  --graphland_different_transform \
  --seeds "${EVAL_SEEDS[@]}" \
  "${EXTRA_ARGS[@]}"

METRICS_FILE="${SAVE_LOAD_PATH}/${RUN_NAME}/metrics.json"
if [[ ! -s "${METRICS_FILE}" ]]; then
  echo "ERROR: eval finished without metrics at ${METRICS_FILE}" >&2
  exit 1
fi

if [[ -f "${COLLECT_EVAL_SCRIPT}" ]]; then
  "$PYTHON" "${COLLECT_EVAL_SCRIPT}" "${EVAL_RUN_DIR}" --logs-dir "${LOGS_DIR}"
  echo "Updated run summary: ${EVAL_RUN_DIR}/eval_summary.txt"
fi

finish_job
