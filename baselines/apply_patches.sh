#!/usr/bin/env bash
# Apply Wander patches on third_party checkouts.
#
#   bash baselines/apply_patches.sh
#   bash baselines/apply_patches.sh --check
#   bash baselines/apply_patches.sh --skip-missing   # Wander-only (no error)

set -euo pipefail

WANDER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${WANDER_ROOT}"

CHECK_ONLY=0
SKIP_MISSING=0
for arg in "$@"; do
  case "${arg}" in
    --check) CHECK_ONLY=1 ;;
    --skip-missing) SKIP_MISSING=1 ;;
  esac
done

apply_one() {
  local repo="$1"
  local patch="$2"
  if [[ ! -d "${repo}/.git" && ! -f "${repo}/.git" ]]; then
    if [[ "${SKIP_MISSING}" -eq 1 ]]; then
      echo "SKIP ${repo}: checkout missing"
      return 0
    fi
    echo "ERROR: missing checkout ${repo}" >&2
    echo "Clone pinned remotes first: bash scripts/setup_third_party.sh" >&2
    return 1
  fi
  if [[ ! -f "${patch}" ]]; then
    echo "ERROR: missing patch ${patch}" >&2
    return 1
  fi
  echo "Applying $(basename "${patch}") -> ${repo}"
  if git -C "${repo}" apply --reverse --check "${WANDER_ROOT}/${patch}" 2>/dev/null; then
    echo "  already applied"
    return 0
  fi
  if [[ "${CHECK_ONLY}" == "1" ]]; then
    git -C "${repo}" apply --check "${WANDER_ROOT}/${patch}"
    echo "  OK (would apply)"
    return 0
  fi
  git -C "${repo}" apply "${WANDER_ROOT}/${patch}"
  echo "  applied"
}

apply_one third_party/graphpfn baselines/graphpfn/patches/graphpfn-wander.patch
apply_one third_party/AnyGraph baselines/anygraph/patches/anygraph-wander-eval.patch
apply_one third_party/GraphAny baselines/graphany/patches/graphany-wander.patch
apply_one third_party/NodePFN baselines/nodepfn/patches/nodepfn-torch2.patch

GRAPHPFN_EXP="${WANDER_ROOT}/third_party/graphpfn/exp"
CONFIGS="${WANDER_ROOT}/baselines/graphpfn/configs"
if [[ -d "${GRAPHPFN_EXP}" && -d "${CONFIGS}" ]]; then
  ln -sfn ../../../baselines/graphpfn/configs "${GRAPHPFN_EXP}/wander"
  echo "Symlink ${GRAPHPFN_EXP}/wander -> baselines/graphpfn/configs"
fi

echo "Done."
