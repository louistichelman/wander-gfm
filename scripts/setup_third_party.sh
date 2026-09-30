#!/usr/bin/env bash
# Clone pinned baseline checkouts into third_party/ and apply Wander patches.
# Needed for baselines, Table 5 prior ablations, official UniLP, Figures 3–4.
# Wander-only tables do not need this.
#
#   bash scripts/setup_third_party.sh
#   bash scripts/setup_third_party.sh graphpfn NodePFN
#   bash scripts/setup_third_party.sh --no-patch

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PINS="${ROOT}/third_party/PINS"
APPLY_PATCHES=1
WANTED=()

for arg in "$@"; do
  case "${arg}" in
    --no-patch) APPLY_PATCHES=0 ;;
    -h|--help)
      sed -n '2,10p' "$0"
      exit 0
      ;;
    *) WANTED+=("${arg}") ;;
  esac
done

if [[ ! -f "${PINS}" ]]; then
  echo "ERROR: missing ${PINS}" >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git is required to clone third_party checkouts" >&2
  exit 1
fi

clone_one() {
  local name="$1" url="$2" sha="$3"
  local dest="${ROOT}/third_party/${name}"
  if [[ -d "${dest}/.git" || -f "${dest}/.git" ]]; then
    echo "=== ${name}: already a git checkout, fetching ${sha} ==="
    git -C "${dest}" fetch --depth 1 origin "${sha}" 2>/dev/null \
      || git -C "${dest}" fetch origin "${sha}"
    git -C "${dest}" checkout --detach "${sha}"
    return 0
  fi
  if [[ -d "${dest}" ]] && [[ -n "$(ls -A "${dest}" 2>/dev/null)" ]]; then
    echo "ERROR: ${dest} exists and is not a git checkout. Move it aside." >&2
    exit 1
  fi
  echo "=== ${name}: clone ${url} @ ${sha} ==="
  mkdir -p "${ROOT}/third_party"
  rm -rf "${dest}"
  git clone --filter=blob:none "${url}" "${dest}"
  git -C "${dest}" checkout --detach "${sha}"
}

while IFS=$'\t' read -r name url sha || [[ -n "${name:-}" ]]; do
  [[ -z "${name}" || "${name}" == \#* ]] && continue
  if [[ ${#WANTED[@]} -gt 0 ]]; then
    local_hit=0
    for w in "${WANTED[@]}"; do
      if [[ "${w}" == "${name}" ]]; then
        local_hit=1
        break
      fi
    done
    [[ "${local_hit}" -eq 1 ]] || continue
  fi
  clone_one "${name}" "${url}" "${sha}"
done < "${PINS}"

if [[ "${APPLY_PATCHES}" -eq 1 ]]; then
  if [[ ${#WANTED[@]} -gt 0 ]]; then
    bash "${ROOT}/baselines/apply_patches.sh" --skip-missing
  else
    bash "${ROOT}/baselines/apply_patches.sh"
  fi
fi

echo "Done. Place official weights as described in third_party/README.md."
