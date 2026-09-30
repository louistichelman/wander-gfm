#!/usr/bin/env bash
# Submit a job script. Uses sbatch when available; otherwise runs locally.
#
#   bash slurm_scripts/submit.sh slurm_scripts/eval/eval_one_dataset.sh CORA
#   SLURM_PARTITION=gpu SLURM_GRES=gpu:1 bash slurm_scripts/submit.sh ...
#
# Queue coordinators (queue_*.sh) always run locally.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash slurm_scripts/submit.sh <job_script.sh> [args...]" >&2
  exit 1
fi

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/slurm_scripts/cluster.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/slurm_scripts"
elif [[ -f "$(dirname "$0")/cluster.sh" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
else
  echo "ERROR: cannot find slurm_scripts/cluster.sh" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/cluster.sh"

JOB_SCRIPT="$1"
shift

if [[ ! -f "${JOB_SCRIPT}" ]]; then
  echo "ERROR: job script not found: ${JOB_SCRIPT}" >&2
  exit 1
fi

JOB_BASE="$(basename "${JOB_SCRIPT}")"
if [[ "${JOB_BASE}" == queue_*.sh || "${JOB_BASE}" == submit_*.sh ]]; then
  echo "Coordinator ${JOB_BASE} → bash (not sbatch)"
  exec bash "${JOB_SCRIPT}" "$@"
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found; running locally: bash ${JOB_SCRIPT} $*"
  exec bash "${JOB_SCRIPT}" "$@"
fi

KIND="$(cluster_job_kind "${JOB_SCRIPT}")"
mapfile -t SBATCH_FLAGS < <(cluster_sbatch_flags "${KIND}")
[[ -n "${SLURM_DEPENDENCY:-}" ]] && SBATCH_FLAGS+=("--dependency=${SLURM_DEPENDENCY}")
[[ -n "${SLURM_JOB_NAME:-}" ]] && SBATCH_FLAGS+=("--job-name=${SLURM_JOB_NAME}")

cmd=(sbatch "${SBATCH_FLAGS[@]}" "${JOB_SCRIPT}" "$@")
printf '%q ' "${cmd[@]}"
echo
"${cmd[@]}"
