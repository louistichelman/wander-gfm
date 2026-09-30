#!/usr/bin/env bash
# Login-node coordinator: nofeat long-range NC (PubMed / Full DBLP / Co CS /
# Co Physics) with Phase B walks that visit a train node only (keep_p=0).
# Samples 2x size-tier walk_num, then drops train-free walks. Same protocol
# as queue_eval_nc_longrange.sh otherwise (no GRIDS).
# Do NOT sbatch this file.
#
# Usage (from the repo root):
#   bash slurm_scripts/eval/queue_eval_nc_longrange_train_hits.sh
#   bash slurm_scripts/eval/queue_eval_nc_longrange_train_hits.sh --dry-run

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: queue_eval_nc_longrange_train_hits.sh is a login-node coordinator — run with bash, not sbatch." >&2
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

export IGNORE_FEATURES=1
export WANDER_KEEP_TRAIN_FREE_P="${WANDER_KEEP_TRAIN_FREE_P:-0}"
if [[ -z "${DATASETS:-}" ]]; then
  DATASETS="PUBMED FULL_DBLP CO_CS CO_PHYSICS"
  export DATASETS
fi
export EVAL_RUN_DIR="${EVAL_RUN_DIR:-${REPO_ROOT}/results/eval/nc_longrange_train_hits_$(date +%y%m%d_%H%M)}"

echo "=== Paper Table 7 long-range NC (nofeat, train-hitting Phase B walks, p=${WANDER_KEEP_TRAIN_FREE_P}) ==="
exec bash "${SLURM_SCRIPTS_DIR}/eval/queue_eval_nc_longrange.sh" "$@"
