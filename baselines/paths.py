"""Filesystem roots for baseline harnesses and third-party checkouts.

Importing this module does not require GraphPFN / NodePFN / AnyGraph / GraphAny /
UniLP (context_LP) to be present. Call :func:`require_checkout` before importing
their code.
"""

from __future__ import annotations

import os
from pathlib import Path

WANDER_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY_ROOT = WANDER_ROOT / "third_party"

GRAPHPFN_ROOT = THIRD_PARTY_ROOT / "graphpfn"
NODEPFN_REPO_ROOT = THIRD_PARTY_ROOT / "NodePFN"
NODEPFN_ROOT = NODEPFN_REPO_ROOT / "nodepfn"
ANYGRAPH_ROOT = THIRD_PARTY_ROOT / "AnyGraph"
GRAPHANY_ROOT = THIRD_PARTY_ROOT / "GraphAny"
CONTEXT_LP_ROOT = THIRD_PARTY_ROOT / "context_LP"
UNILP_ROOT = CONTEXT_LP_ROOT

GRAPHPFN_CONFIG_ROOT = WANDER_ROOT / "baselines" / "graphpfn" / "configs"
GRAPHPFN_PATCH = WANDER_ROOT / "baselines" / "graphpfn" / "patches" / "graphpfn-wander.patch"
ANYGRAPH_PATCH = WANDER_ROOT / "baselines" / "anygraph" / "patches" / "anygraph-wander-eval.patch"
GRAPHANY_PATCH = WANDER_ROOT / "baselines" / "graphany" / "patches" / "graphany-wander.patch"

_CHECKOUTS: dict[str, Path] = {
    "graphpfn": GRAPHPFN_ROOT,
    "nodepfn": NODEPFN_REPO_ROOT,
    "anygraph": ANYGRAPH_ROOT,
    "graphany": GRAPHANY_ROOT,
    "unilp": CONTEXT_LP_ROOT,
    "context_lp": CONTEXT_LP_ROOT,
}

_CLONE_HINTS: dict[str, str] = {
    "graphpfn": (
        "git submodule update --init third_party/graphpfn && "
        "bash baselines/apply_patches.sh"
    ),
    "nodepfn": (
        "git submodule update --init third_party/NodePFN && "
        "bash baselines/apply_patches.sh"
    ),
    "anygraph": (
        "git submodule update --init third_party/AnyGraph && "
        "bash baselines/apply_patches.sh"
    ),
    "graphany": (
        "git submodule update --init third_party/GraphAny && "
        "bash baselines/apply_patches.sh"
    ),
    "unilp": "git submodule update --init third_party/context_LP",
    "context_lp": "git submodule update --init third_party/context_LP",
}


def checkout_present(name: str) -> bool:
    """Return True if ``name`` looks like an initialized third-party checkout."""
    path = _CHECKOUTS[name]
    if not path.is_dir():
        return False
    git_dir = path / ".git"
    if git_dir.exists():
        return True
    return (path / "README.md").is_file() or (path / "main.py").is_file()


def require_checkout(name: str) -> Path:
    """Return the checkout path, or raise with clone/submodule instructions."""
    if name not in _CHECKOUTS:
        raise KeyError(f"Unknown third-party checkout {name!r}")
    path = _CHECKOUTS[name]
    if checkout_present(name):
        return path
    hint = _CLONE_HINTS[name]
    raise FileNotFoundError(
        f"Missing third-party checkout {name!r} at {path}.\n"
        f"From the Wander repo root, run:\n  {hint}"
    )


def resolve_context_lp_root() -> Path:
    """Official UniLP / context_LP checkout.

    Preference: ``CONTEXT_LP_ROOT`` / ``UNILP_ROOT`` env, then
    ``third_party/context_LP``, then a leftover sibling
    ``<Wander>/../context_LP``.
    """
    for key in ("CONTEXT_LP_ROOT", "UNILP_ROOT"):
        env = os.environ.get(key, "").strip()
        if env:
            return Path(env)
    if checkout_present("unilp"):
        return CONTEXT_LP_ROOT
    sibling = WANDER_ROOT.parent / "context_LP"
    if (sibling / "main.py").is_file() or (sibling / "snap_dataset.py").is_file():
        return sibling
    return CONTEXT_LP_ROOT


def resolve_unilp_data_root() -> Path:
    """Folder official UniLP and Wander UniLP loaders treat as ``dataset_dir``.

    Preference: ``UNILP_DATA_DIR``, then ``third_party/context_LP/data``, then
    Wander ``raw_data`` (mats under ``raw_data/UniLP/``), then a sibling
    ``../context_LP/data``.
    """
    env = os.environ.get("UNILP_DATA_DIR", "").strip()
    if env:
        return Path(env)
    checkout_data = resolve_context_lp_root() / "data"
    if checkout_data.is_dir() and (
        any(checkout_data.glob("*.mat")) or (checkout_data / "UniLP").is_dir()
    ):
        return checkout_data
    wander = WANDER_ROOT / "raw_data"
    if any((wander / "UniLP").glob("*.mat")) or any(wander.glob("*.mat")):
        return wander
    sibling = WANDER_ROOT.parent / "context_LP" / "data"
    if sibling.is_dir():
        return sibling
    return checkout_data


def resolve_unilp_checkpoint() -> Path:
    """Released UniLP weights ``checkpoints/pretrained/model.pt``."""
    env = os.environ.get("UNILP_CKPT", "").strip()
    if env:
        return Path(env)
    return resolve_context_lp_root() / "checkpoints" / "pretrained" / "model.pt"


def resolve_anygraph_data_root() -> Path:
    """Pickle root used by patched AnyGraph ``data_handler.py``.

    Preference: ``ANYGRAPH_DATA_ROOT`` env, then Wander's extracted archive at
    ``raw_data/AnyGraph/zero-shot datasets``, then the checkout-local folder.
    """
    env = os.environ.get("ANYGRAPH_DATA_ROOT", "").strip()
    if env:
        return Path(env)
    wander = WANDER_ROOT / "raw_data" / "AnyGraph" / "zero-shot datasets"
    if wander.is_dir():
        return wander
    return ANYGRAPH_ROOT / "zero-shot datasets"


def resolve_anygraph_nc_data_root() -> Path:
    """Root whose ``node_data/<slug>/`` pickles AnyGraph NC reads.

    Preference: ``ANYGRAPH_NC_DATA_ROOT``, then ``raw_data/anygraph_nc``.
    Isolated from official AnyGraph LP pickles.
    """
    env = os.environ.get("ANYGRAPH_NC_DATA_ROOT", "").strip()
    if env:
        return Path(env)
    return WANDER_ROOT / "raw_data" / "anygraph_nc"


def resolve_graphany_data_root() -> Path:
    """Npy/npz bundles for GraphAny paper-split eval.

    Preference: ``GRAPHANY_DATA_ROOT``, then ``raw_data/graphany_nc``.
    """
    env = os.environ.get("GRAPHANY_DATA_ROOT", "").strip()
    if env:
        return Path(env)
    return WANDER_ROOT / "raw_data" / "graphany_nc"


def resolve_nodepfn_data_root() -> Path:
    """Npy/npz bundles for NodePFN paper-split eval.

    Preference: ``NODEPFN_DATA_ROOT``, then ``raw_data/nodepfn_nc``.
    """
    env = os.environ.get("NODEPFN_DATA_ROOT", "").strip()
    if env:
        return Path(env)
    return WANDER_ROOT / "raw_data" / "nodepfn_nc"


def resolve_tabpfn_data_root() -> Path:
    """Npy/npz root for TabPFN tabular NC eval (ignore graph).

    Preference: ``TABPFN_DATA_ROOT``, then ``raw_data/tabpfn_nc``.
    Prefer :func:`resolve_tabpfn_bundle_dir` when resolving one slug so
    GraphLand ordinal exports under ``tabpfn_nc`` can coexist with reused
    ``nodepfn_nc`` bundles.
    """
    env = os.environ.get("TABPFN_DATA_ROOT", "").strip()
    if env:
        return Path(env)
    return WANDER_ROOT / "raw_data" / "tabpfn_nc"


def resolve_tabpfn_bundle_dir(slug: str) -> Path:
    """Resolve ``features.npy`` / splits dir for one TabPFN NC slug.

    Order: ``$TABPFN_DATA_ROOT/<slug>``, ``raw_data/tabpfn_nc/<slug>`` if
    present, else ``raw_data/nodepfn_nc/<slug>``.
    """
    env = os.environ.get("TABPFN_DATA_ROOT", "").strip()
    if env:
        return Path(env) / slug
    tabpfn = WANDER_ROOT / "raw_data" / "tabpfn_nc" / slug
    if (tabpfn / "features.npy").is_file():
        return tabpfn
    nodepfn = WANDER_ROOT / "raw_data" / "nodepfn_nc" / slug
    if (nodepfn / "features.npy").is_file():
        return nodepfn
    return tabpfn
