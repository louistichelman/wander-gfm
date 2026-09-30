#!/usr/bin/env bash
# Login-node coordinator: paper Table 6 MiniLM 2×2 composition.
# Features on/off × relations on/off on ConceptNet100k, WN18RR, WN-Ind v1–v4.
# Feat cells use PCA-32 MiniLM keys; structure cells use the twins without
# features. Relations off maps every type (incl. inverses) to one dummy type
# via --ignore_edge_types. Same Flock n/P as queue_eval_kg.sh.
# Do NOT sbatch this file.
#
# Usage (from the repo root):
#   bash slurm_scripts/eval/queue_eval_kg_composition.sh
#   bash slurm_scripts/eval/queue_eval_kg_composition.sh --dry-run
#   CELL=feat_norel bash slurm_scripts/eval/queue_eval_kg_composition.sh

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: queue_eval_kg_composition.sh is a login-node coordinator — run with bash, not sbatch." >&2
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

CELL="${CELL:-all}"
ROOT="${EVAL_RUN_DIR:-${REPO_ROOT}/results/eval/kg_composition_$(date +%y%m%d_%H%M)}"
mkdir -p "${ROOT}" logs

should_run() {
  local name="$1"
  [[ "${CELL}" == "all" || "${CELL}" == "${name}" ]]
}

run_cell() {
  local tag="$1"
  local datasets="$2"
  local ignore_edge_types="$3"
  echo "=== ${tag} (ignore_edge_types=${ignore_edge_types}) ==="
  env \
    "DATASETS=${datasets}" \
    "EVAL_RUN_DIR=${ROOT}/${tag}" \
    "IGNORE_FEATURES=0" \
    "IGNORE_EDGE_TYPES=${ignore_edge_types}" \
    "JOB_NAME_PREFIX=ev-kg-${tag}" \
    bash "${SLURM_SCRIPTS_DIR}/eval/queue_eval_kg.sh" "${DRY_ARGS[@]+"${DRY_ARGS[@]}"}"
}

FEAT_DS="${PAPER_KG_FEAT_DATASETS[*]}"
STRUCT_DS="${PAPER_KG_COMPOSITION_STRUCT_DATASETS[*]}"

echo "=== Paper Table 6 MiniLM 2×2 (features × relations) ==="
echo "Repo: ${REPO_ROOT}"
echo "Results: ${ROOT}"
echo "Cell: ${CELL}"

should_run feat_rel && run_cell feat_rel "${FEAT_DS}" 0
should_run struct_rel && run_cell struct_rel "${STRUCT_DS}" 0
should_run feat_norel && run_cell feat_norel "${FEAT_DS}" 1
should_run struct_norel && run_cell struct_norel "${STRUCT_DS}" 1

echo "=== Done queueing KG composition ==="
echo "Results: ${ROOT}"
