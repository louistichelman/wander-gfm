#!/usr/bin/env bash
# Resolve and source common.sh from any nested job script.

# shellcheck source=/dev/null
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

_common="${SLURM_SCRIPTS_DIR}/common.sh"
if [[ ! -f "${_common}" ]]; then
  echo "ERROR: cannot find common.sh (SLURM_SCRIPTS_DIR=${SLURM_SCRIPTS_DIR:-unset})" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${_common}"
