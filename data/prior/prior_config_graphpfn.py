"""Native GraphPFN prior marker for SCMPrior.

When ``native_backend == "graphpfn"``, ``SCMPrior.get_batch`` samples from
GraphPFN's official prior (see ``native_backends.py``) instead of the SCM path.
Complexity is ignored. Node classification skips regression tasks. Link
prediction reuses those graphs as undirected label-stripped LP.
"""

from .prior_config_base import build_get_default_fixed_hp, build_get_default_sampled_hp


def _graphpfn_fixed_hp_dict(complexity: float) -> dict:
    del complexity  # Native GraphPFN pretrain config is complexity-independent.
    return {
        "native_backend": "graphpfn",
    }


def _graphpfn_sampled_hp_dict(complexity: float) -> dict:
    del complexity
    return {}


get_default_fixed_hp = build_get_default_fixed_hp(_graphpfn_fixed_hp_dict)
get_default_sampled_hp = build_get_default_sampled_hp(_graphpfn_sampled_hp_dict)

DEFAULT_SAMPLED_HP = get_default_sampled_hp()
