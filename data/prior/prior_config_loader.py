"""Registry mapping ``--prior_config`` names to HP config modules.

``default`` is Wander's SCM prior; ``graphpfn`` / ``nodepfn`` select the
native backends in ``native_backends.py``.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Literal

PriorConfigName = Literal["default", "graphpfn", "nodepfn"]

_REGISTRY: dict[str, str] = {
    "default": "data.prior.prior_config",
    "graphpfn": "data.prior.prior_config_graphpfn",
    "nodepfn": "data.prior.prior_config_nodepfn",
}

PRIOR_CONFIG_NAMES: tuple[str, ...] = tuple(_REGISTRY)
NATIVE_PRIOR_CONFIGS: frozenset[str] = frozenset({"graphpfn", "nodepfn"})


def load_prior_config(name: str) -> ModuleType:
    """Import and return the prior config module for ``name``."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown prior_config {name!r}; expected one of {sorted(_REGISTRY)}"
        )
    return importlib.import_module(_REGISTRY[name])


def prior_config_module_path(name: str) -> str:
    """Return the filesystem path of the module source for checkpoint snapshots."""
    mod = load_prior_config(name)
    return mod.__file__
