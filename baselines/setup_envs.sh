#!/usr/bin/env bash
# Create baseline conda envs. Do not mix these with wander.
#
#   bash baselines/setup_envs.sh
#   bash baselines/setup_envs.sh graphany_env
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
#     bash baselines/setup_envs.sh graphpfn_env
#
# PyTorch wheels come from TORCH_INDEX_URL (default cu124). Install a DGL
# build that matches that wheel (DGL_WHEEL_URL or the DGL install matrix).

set -euo pipefail

WANDER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${WANDER_ROOT}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Create the envs with conda or micromamba, then pip-install." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

create_env() {
  local name="$1"
  local py="$2"
  if conda env list | awk '{print $1}' | grep -qx "${name}"; then
    echo "=== ${name} already exists ==="
    return 0
  fi
  echo "=== conda create -n ${name} python=${py} ==="
  conda create -y -n "${name}" "python=${py}" pip
}

pip_in() {
  local name="$1"
  shift
  conda run -n "${name}" --no-capture-output python -m pip install --upgrade pip
  conda run -n "${name}" --no-capture-output python -m pip install "$@"
}

install_torch() {
  local name="$1"
  pip_in "${name}" torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"
}

install_dgl() {
  local name="$1"
  if [[ -n "${DGL_WHEEL_URL:-}" ]]; then
    echo "=== ${name}: DGL from DGL_WHEEL_URL ==="
    pip_in "${name}" dgl -f "${DGL_WHEEL_URL}"
    return 0
  fi
  echo "=== ${name}: DGL not installed (set DGL_WHEEL_URL or install a matching DGL wheel) ==="
  echo "    https://www.dgl.ai/pages/start.html"
}

setup_graphany_env() {
  create_env graphany_env 3.12
  install_torch graphany_env
  install_dgl graphany_env
  pip_in graphany_env -r "${WANDER_ROOT}/baselines/graphany/requirements.txt"
}

setup_graphpfn_env() {
  create_env graphpfn_env 3.12
  install_torch graphpfn_env
  install_dgl graphpfn_env
  pip_in graphpfn_env -r "${WANDER_ROOT}/baselines/graphpfn/requirements.txt"
}

setup_anygraph_env() {
  create_env anygraph_env 3.10
  install_torch anygraph_env
  pip_in anygraph_env -r "${WANDER_ROOT}/baselines/anygraph/requirements.txt"
}

setup_unilp_env() {
  create_env unilp_env 3.10
  install_torch unilp_env
  pip_in unilp_env -r "${WANDER_ROOT}/baselines/unilp/requirements.txt"
}

setup_tabpfn_env() {
  create_env tabpfn_env 3.12
  install_torch tabpfn_env
  pip_in tabpfn_env -r "${WANDER_ROOT}/baselines/tabpfn/requirements.txt"
}

setup_nodepfn_env() {
  create_env nodepfn_env 3.11
  install_torch nodepfn_env
  pip_in nodepfn_env -r "${WANDER_ROOT}/baselines/nodepfn/requirements.txt"
}

smoke() {
  local name="$1"
  local code="$2"
  echo "=== smoke ${name} ==="
  conda run -n "${name}" --no-capture-output python -c "${code}"
  echo "OK ${name}"
}

WANTED=("$@")
if [[ ${#WANTED[@]} -eq 0 ]]; then
  WANTED=(graphany_env graphpfn_env anygraph_env nodepfn_env unilp_env tabpfn_env)
fi

for name in "${WANTED[@]}"; do
  case "${name}" in
    graphany_env) setup_graphany_env ;;
    graphpfn_env) setup_graphpfn_env ;;
    anygraph_env) setup_anygraph_env ;;
    nodepfn_env) setup_nodepfn_env ;;
    unilp_env) setup_unilp_env ;;
    tabpfn_env) setup_tabpfn_env ;;
    *) echo "ERROR: unknown env ${name}" >&2; exit 1 ;;
  esac
done

for name in "${WANTED[@]}"; do
  case "${name}" in
    graphany_env)
      smoke graphany_env "import torch, torch_geometric; print('torch', torch.__version__)" ;;
    graphpfn_env)
      smoke graphpfn_env "import torch, torch_geometric; print('torch', torch.__version__)" ;;
    anygraph_env)
      smoke anygraph_env "import torch, torch_geometric; print('torch', torch.__version__)" ;;
    nodepfn_env)
      smoke nodepfn_env "import torch, torch_geometric, gpytorch; print('torch', torch.__version__)" ;;
    unilp_env)
      smoke unilp_env "import torch, torch_geometric, ogb; print('torch', torch.__version__)" ;;
    tabpfn_env)
      smoke tabpfn_env "import torch, tabpfn; print('torch', torch.__version__)" ;;
  esac
done

echo "Done. conda env list should show: ${WANTED[*]}"
conda env list
