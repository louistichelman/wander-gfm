"""Native NodePFN prior marker for SCMPrior.

When ``native_backend == "nodepfn"``, ``SCMPrior.get_batch`` samples from the
NodePFN prior stack (see ``native_backends.py``) instead of the SCM path.
Complexity is ignored. Link prediction reuses those graphs as undirected
label-stripped LP.
"""

from .prior_config_base import build_get_default_fixed_hp, build_get_default_sampled_hp


def _nodepfn_fixed_hp_dict(complexity: float) -> dict:
    del complexity  # Native NodePFN pretrain config is complexity-independent.
    return {
        "native_backend": "nodepfn",
    }


def _nodepfn_sampled_hp_dict(complexity: float) -> dict:
    del complexity
    return {}


get_default_fixed_hp = build_get_default_fixed_hp(_nodepfn_fixed_hp_dict)
get_default_sampled_hp = build_get_default_sampled_hp(_nodepfn_sampled_hp_dict)

DEFAULT_SAMPLED_HP = get_default_sampled_hp()
