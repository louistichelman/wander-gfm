#!/usr/bin/env bash
# Login-node coordinator: Table 6 feat+rel cell — ConceptNet + WN18RR /
# WN-Ind v1–v4 with precomputed PCA-32 MiniLM node features.
# The full features × relations 2×2 is queue_eval_kg_composition.sh.
# Same Flock n/P as the structure-only twins (kg_protocol.sh).
# Do NOT sbatch this file.
#
# Usage (from the repo root):
#   bash slurm_scripts/eval/queue_eval_kg_feat.sh
#   bash slurm_scripts/eval/queue_eval_kg_feat.sh --dry-run

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: queue_eval_kg_feat.sh is a login-node coordinator — run with bash, not sbatch." >&2
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

if [[ -z "${DATASETS:-}" ]]; then
  DATASETS="${PAPER_KG_FEAT_DATASETS[*]}"
  export DATASETS
fi
export EVAL_RUN_DIR="${EVAL_RUN_DIR:-${REPO_ROOT}/results/eval/kg_feat_$(date +%y%m%d_%H%M)}"

echo "=== Paper MiniLM KG composition (ConceptNet + WN feat) ==="
exec bash "${SLURM_SCRIPTS_DIR}/eval/queue_eval_kg.sh" "$@"
