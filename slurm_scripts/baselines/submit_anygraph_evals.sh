#!/bin/bash
# Queue paper-style AnyGraph zero-shot jobs.
#
# Usage (from the repo root, login node):
#   bash slurm_scripts/baselines/submit_anygraph_evals.sh
#   EVAL_PROTOCOL_WANDER=1 bash slurm_scripts/baselines/submit_anygraph_evals.sh

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: do not sbatch this file; run it on the login node." >&2
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
SCRIPT_DIR="${SLURM_SCRIPTS_DIR}"
cd "${REPO_ROOT}"
JOB_SCRIPT="${SLURM_SCRIPTS_DIR}/baselines/anygraph_eval.sh"
EVAL_PROTOCOL_WANDER="${EVAL_PROTOCOL_WANDER:-0}"

mkdir -p logs

submit_eval() {
  local load_model="$1"
  local dataset_setting="$2"
  local eval_type="$3"
  local job_name="ag-${load_model}-${dataset_setting}"
  if [[ "${EVAL_PROTOCOL_WANDER}" == "1" ]]; then
    job_name="${job_name}-wander"
  fi
  echo "Submitting ${job_name} (${eval_type})"
  EVAL_PROTOCOL_WANDER="${EVAL_PROTOCOL_WANDER}" \
    SLURM_JOB_NAME="${job_name}" \
    bash slurm_scripts/submit.sh \
      "${JOB_SCRIPT}" \
      "${load_model}" \
      "${dataset_setting}" \
      "${eval_type}"
}

submit_eval pretrain_link1 link2 link
submit_eval pretrain_link2 link1 link
submit_eval pretrain_link1 ecommerce_in_link2 link
submit_eval pretrain_link2 ecommerce_in_link1 link
submit_eval pretrain_link1 academic_in_link2 link
submit_eval pretrain_link2 academic_in_link1 link
submit_eval pretrain_link1 others_in_link2 link
submit_eval pretrain_link2 others_in_link1 link
submit_eval pretrain_link2 node node
