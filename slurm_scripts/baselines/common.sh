#!/usr/bin/env bash
# Shared setup for GraphPFN / NodePFN / GraphAny / AnyGraph / UniLP / TabPFN jobs
# (does not require the Wander conda env).
#
# In-repo GCN / BUDDY / NBFNet / heuristic jobs in this folder must source
# slurm_scripts/source_common.sh (parent) instead — that helper activates
# wander.

set -euo pipefail

SLURM_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${SLURM_SCRIPT_DIR}/_lib.sh"
REPO_ROOT="$(cd "${SLURM_SCRIPT_DIR}/.." && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/baselines/paths.py" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi
cd "${REPO_ROOT}"

THIRD_PARTY="${REPO_ROOT}/third_party"
GRAPHPFN_ROOT="${THIRD_PARTY}/graphpfn"
NODEPFN_ROOT="${THIRD_PARTY}/NodePFN"
GRAPHANY_ROOT="${THIRD_PARTY}/GraphAny"
ANYGRAPH_ROOT="${THIRD_PARTY}/AnyGraph"
CONTEXT_LP_ROOT="${THIRD_PARTY}/context_LP"
LOGS_DIR="${LOGS_DIR:-${REPO_ROOT}/logs}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results/baselines}"
PYTHON="${PYTHON:-python}"
WANDER_DATA_DIR="${WANDER_DATA_DIR:-${REPO_ROOT}/raw_data}"
WANDER_ENV="${WANDER_ENV:-wander}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export DGLBACKEND="${DGLBACKEND:-pytorch}"

mkdir -p "${LOGS_DIR}" "${RESULTS_DIR}"

echo "==========================================="
echo "Baseline job started: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Repo: ${REPO_ROOT}"
echo "==========================================="
