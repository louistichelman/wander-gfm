#!/usr/bin/env bash
# Locate slurm_scripts/ and the repo root (directory that contains main.py).

if [[ -n "${SLURM_SCRIPTS_DIR:-}" && -f "${SLURM_SCRIPTS_DIR}/_lib.sh" ]]; then
  REPO_ROOT="${REPO_ROOT:-$(cd "${SLURM_SCRIPTS_DIR}/.." && pwd)}"
  return 0 2>/dev/null || true
fi

_find_slurm_scripts() {
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/slurm_scripts/_lib.sh" ]]; then
    printf '%s' "${SLURM_SUBMIT_DIR}/slurm_scripts"
    return 0
  fi
  local d
  d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -f "${d}/_lib.sh" && -f "${d}/common.sh" ]]; then
    printf '%s' "${d}"
    return 0
  fi
  d="$(cd "$(dirname "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")" && pwd)"
  while [[ "${d}" != "/" ]]; do
    if [[ -f "${d}/_lib.sh" && -f "${d}/common.sh" ]]; then
      printf '%s' "${d}"
      return 0
    fi
    d="$(dirname "${d}")"
  done
  return 1
}

if ! SLURM_SCRIPTS_DIR="$(_find_slurm_scripts)"; then
  echo "ERROR: cannot locate slurm_scripts/_lib.sh" >&2
  echo "Run from the repo root: bash slurm_scripts/submit.sh slurm_scripts/<script>.sh" >&2
  return 1 2>/dev/null || exit 1
fi
REPO_ROOT="$(cd "${SLURM_SCRIPTS_DIR}/.." && pwd)"
unset -f _find_slurm_scripts

maybe_load_conda_module() {
  command -v conda >/dev/null 2>&1 && return 0
  if [[ -n "${CONDA_MODULE:-}" ]] && command -v module >/dev/null 2>&1; then
    module load "${CONDA_MODULE}" 2>/dev/null || true
  fi
}

find_env_python() {
  local env_name="$1"
  local override="${2:-}"
  local root candidate base prefix
  if [[ -n "${override}" && -x "${override}" ]]; then
    printf '%s' "${override}"
    return 0
  fi

  local roots=()
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    prefix="${CONDA_PREFIX}"
    if [[ "${prefix}" == */envs/* ]]; then
      roots+=("${prefix%/envs/*}")
    else
      roots+=("${prefix}")
    fi
  fi
  if command -v conda >/dev/null 2>&1; then
    base="$(conda info --base 2>/dev/null)" || base=""
    [[ -n "${base}" ]] && roots+=("${base}")
    prefix="$(conda env list 2>/dev/null | awk -v n="${env_name}" '$1 == n { print $NF; exit }')" || prefix=""
    if [[ -n "${prefix}" && -x "${prefix}/bin/python" ]]; then
      printf '%s' "${prefix}/bin/python"
      return 0
    fi
  fi
  roots+=(
    "${HOME}/miniconda3"
    "${HOME}/miniforge3"
    "${HOME}/mambaforge"
    "${HOME}/anaconda3"
    "${MAMBA_ROOT_PREFIX:-${HOME}/micromamba}"
  )

  local seen="|"
  for root in "${roots[@]}"; do
    [[ -z "${root}" ]] && continue
    [[ "${seen}" == *"|${root}|"* ]] && continue
    seen+="${root}|"
    for candidate in \
      "${root}/envs/${env_name}/bin/python" \
      "${root}/${env_name}/bin/python"
    do
      if [[ -x "${candidate}" ]]; then
        printf '%s' "${candidate}"
        return 0
      fi
    done
  done
  return 1
}

require_env_python() {
  local env_name="$1"
  local override="${2:-}"
  local py
  maybe_load_conda_module
  if py="$(find_env_python "${env_name}" "${override}")"; then
    printf '%s' "${py}"
    return 0
  fi
  echo "ERROR: cannot find Python for conda env '${env_name}'." >&2
  echo "Activate that env, or set SUBMIT_PYTHON / the job's *_PYTHON override." >&2
  return 1
}

default_unilp_data_dir() {
  if [[ -n "${UNILP_DATA_DIR:-}" ]]; then
    printf '%s' "${UNILP_DATA_DIR}"
    return 0
  fi
  if [[ -d "${REPO_ROOT}/third_party/context_LP/data" ]]; then
    printf '%s' "${REPO_ROOT}/third_party/context_LP/data"
    return 0
  fi
  if [[ -d "${REPO_ROOT}/raw_data/UniLP" ]]; then
    printf '%s' "${REPO_ROOT}/raw_data"
    return 0
  fi
  printf '%s' "${REPO_ROOT}/third_party/context_LP/data"
}
