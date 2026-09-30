#!/usr/bin/env bash
# Login-node coordinator: paper Table 16 UniLP vs Wander sampled Recall@20.
# Seven graphs, two 70/10/20 splits, 100 sampled non-edges per source.
# Nofeat on all seven; featured Wander on Cora / CS / Facebook.
# Do NOT sbatch this file.
#
# Usage (from the repo root):
#   bash slurm_scripts/eval/inference_ablations/queue_eval_sampled_recall.sh
#   bash slurm_scripts/eval/inference_ablations/queue_eval_sampled_recall.sh --dry-run
#   DATASETS="UNILP_USAIR UNILP_NS" bash slurm_scripts/eval/inference_ablations/queue_eval_sampled_recall.sh

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: queue_eval_sampled_recall.sh is a login-node coordinator — run with bash, not sbatch." >&2
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
source "${_slurm}/_lib.sh"
# shellcheck source=/dev/null
source "${SLURM_SCRIPTS_DIR}/paper_protocol.sh"

export LINK_PRED_EVAL="${LINK_PRED_EVAL:-sampled_recall}"
export SAMPLED_RECALL_NUM_NEG="${SAMPLED_RECALL_NUM_NEG:-100}"
export EVAL_SEEDS="${EVAL_SEEDS:-0:1}"
export UNILP_REQUIRE_EXISTING_SPLITS="${UNILP_REQUIRE_EXISTING_SPLITS:-0}"

ROOT="${EVAL_RUN_DIR:-${REPO_ROOT}/results/eval/unilp_table19_$(date +%y%m%d_%H%M)}"
UNILP_QUEUE="${SLURM_SCRIPTS_DIR}/eval/queue_eval_unilp.sh"

if [[ -z "${DATASETS:-}" ]]; then
  NOFEAT_DATASETS="${PAPER_UNILP_DATASETS[*]}"
else
  NOFEAT_DATASETS="${DATASETS}"
fi

echo "=== Paper Table 16 sampled Recall@20 (2 splits, 100 negs) ==="
echo "Nofeat: ${NOFEAT_DATASETS}"
echo "Feat: ${PAPER_UNILP_FEATURED_DATASETS[*]}"

DATASETS="${NOFEAT_DATASETS}" \
  IGNORE_FEATURES=1 \
  EVAL_RUN_DIR="${ROOT}/nofeat" \
  bash "${UNILP_QUEUE}" "$@"

if [[ -z "${DATASETS:-}" || "${DATASETS}" == "${PAPER_UNILP_DATASETS[*]}" ]]; then
  DATASETS="${PAPER_UNILP_FEATURED_DATASETS[*]}" \
    IGNORE_FEATURES=0 \
    EVAL_RUN_DIR="${ROOT}/feat" \
    bash "${UNILP_QUEUE}" "$@"
fi
