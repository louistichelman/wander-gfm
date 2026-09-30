#!/usr/bin/env bash
# Login-node coordinator: paper Tables 7 / 20 / 21 (Q4) inference-time scaling.
# Default Appendix D / E.6 protocol, then independently:
#   1/2× and 1/4× K_base (walk length unchanged; NC context + query passes)
#   1/2× and 1/4× walk length (K_base unchanged)
#   ensembles 8 and 1 (default walks, WANDER_TEST_SAMPLES)
# NC / LP = paper reporting sets from paper_protocol.sh.
# Do NOT sbatch this file.
#
# Usage (from the repo root):
#   bash slurm_scripts/eval/inference_ablations/queue_eval_inference_scaling.sh
#   bash slurm_scripts/eval/inference_ablations/queue_eval_inference_scaling.sh --dry-run
#   TASK=nc bash slurm_scripts/eval/inference_ablations/queue_eval_inference_scaling.sh
#   TASK=lp VARIANT=walks_half bash slurm_scripts/eval/inference_ablations/queue_eval_inference_scaling.sh
#   TASK=nc VARIANT=len_half bash slurm_scripts/eval/inference_ablations/queue_eval_inference_scaling.sh

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: queue_eval_inference_scaling.sh is a login-node coordinator — run with bash, not sbatch." >&2
  exit 1
fi

DRY_ARGS=()
if [[ "${1:-}" == "--dry-run" || "${1:-}" == "-n" ]]; then
  DRY_ARGS=("--dry-run")
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
cd "${REPO_ROOT}"

TASK="${TASK:-all}"
VARIANT="${VARIANT:-all}"
ROOT="${EVAL_RUN_DIR:-${REPO_ROOT}/results/eval/inference_scaling_$(date +%y%m%d_%H%M)}"
mkdir -p "${ROOT}" logs

run_nc() {
  local tag="$1"
  shift
  echo "=== NC ${tag} ==="
  env \
    "DATASETS=${DATASETS:-${PAPER_NC_DATASETS[*]}}" \
    "EVAL_RUN_DIR=${ROOT}/nc/${tag}" \
    "WANDER_TEST_SAMPLES=${WANDER_TEST_SAMPLES:-16}" \
    "$@" \
    bash "${SLURM_SCRIPTS_DIR}/eval/queue_eval_nc.sh" "${DRY_ARGS[@]+"${DRY_ARGS[@]}"}"
}

run_lp() {
  local tag="$1"
  shift
  echo "=== LP ${tag} ==="
  env \
    "DATASETS=${DATASETS:-${PAPER_LP_DATASETS[*]}}" \
    "EVAL_RUN_DIR=${ROOT}/lp/${tag}" \
    "WANDER_TEST_SAMPLES=${WANDER_TEST_SAMPLES:-16}" \
    "$@" \
    bash "${SLURM_SCRIPTS_DIR}/eval/queue_eval_lp.sh" "${DRY_ARGS[@]+"${DRY_ARGS[@]}"}"
}

should_run() {
  local name="$1"
  [[ "${VARIANT}" == "all" || "${VARIANT}" == "${name}" ]]
}

echo "=== Paper Tables 7/20/21 inference-time scaling ==="
echo "Repo: ${REPO_ROOT}"
echo "Results: ${ROOT}"
echo "Task: ${TASK}  variant: ${VARIANT}"

if [[ "${TASK}" == "all" || "${TASK}" == "nc" ]]; then
  should_run default && run_nc default
  should_run walks_half && run_nc walks_half SIZE_TIER_WALK_NUM_DIV=2
  should_run walks_quarter && run_nc walks_quarter SIZE_TIER_WALK_NUM_DIV=4
  should_run len_half && run_nc len_half SIZE_TIER_WALK_LEN_DIV=2
  should_run len_quarter && run_nc len_quarter SIZE_TIER_WALK_LEN_DIV=4
  should_run ens8 && run_nc ens8 WANDER_TEST_SAMPLES=8
  should_run ens1 && run_nc ens1 WANDER_TEST_SAMPLES=1
fi

if [[ "${TASK}" == "all" || "${TASK}" == "lp" ]]; then
  should_run default && run_lp default
  should_run walks_half && run_lp walks_half WANDER_WALK_NUM_SCALE=2
  should_run walks_quarter && run_lp walks_quarter WANDER_WALK_NUM_SCALE=4
  should_run len_half && run_lp len_half WANDER_WALK_LEN_SCALE=2
  should_run len_quarter && run_lp len_quarter WANDER_WALK_LEN_SCALE=4
  should_run ens8 && run_lp ens8 WANDER_TEST_SAMPLES=8
  should_run ens1 && run_lp ens1 WANDER_TEST_SAMPLES=1
fi

echo "=== Done queueing inference scaling ==="
echo "Results: ${ROOT}"
