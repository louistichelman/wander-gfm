"""Identify Wander state-dict keys for the random-walk update path (checkpoint merge / reset).

Kept free of ``torch_geometric`` imports. Used by training (:mod:`experiment.experiment`).
"""
from __future__ import annotations

import warnings
from typing import Dict, Mapping, Optional, Set, Tuple

import torch

# Legacy checkpoints used a single shared node init vector; current Wander uses a
# 3-row role embedding (other / query-neighbor / query). Row 0 matches the old init.
_NODE_INIT_KEY = "node_init"
_NODE_ROLE_EMB_KEY = "node_role_emb.weight"

# Prefixes match ``Wander`` Phase 2c (RW path) plus walk-embedding gates.
# Exact entries (no trailing dot) match that key or ``name.startswith(entry + ".")``.
_RW_STATE_DICT_PREFIXES: Tuple[str, ...] = (
    "emb_anon_node.",
    "emb_anon_type.",
    "emb_neighbor.",
    "emb_direction.",
    "emb_node_is_query.",
    "emb_type_is_query.",
    "net_rw.",
    "from_node.",
    "from_feature.",
    "from_label.",
    "from_type.",
    "to_node.",
    "to_feature.",
    "to_label.",
    "to_type.",
    "to_type_feature.",
    "to_type_label.",
    "node_logit.",
    "feature_logit.",
    "label_logit.",
    "type_logit.",
    "type_logit_feature.",
    "type_logit_label.",
    "walk_emb_gate_node",
    "walk_emb_gate_feature",
    "walk_emb_gate_label",
    "pre_norm_node.",
    "pre_norm_type.",
    "pre_norm_feature.",
    "pre_norm_label.",
)

# LayerScale on RW scatter residuals only (see ``wander.py`` additive merge after RW).
_LS_EXACT: frozenset[str] = frozenset({"ls_node", "ls_type", "ls_feature", "ls_label"})

# Removed in label-encoding / PE simplification; ignored when loading old checkpoints.
_OBSOLETE_CHECKPOINT_PREFIXES: Tuple[str, ...] = (
    "feature_discriminator.",
    "label_discriminator.",
    "node_cls_head.",
    "label_int_embedding.",
    "emb_label_unknown.",
    "emb_restart.",
)


def filter_obsolete_checkpoint_keys(state_dict: Mapping[str, object]) -> Dict[str, object]:
    """Drop checkpoint tensors for modules removed from the current Wander model."""
    return {
        k: v
        for k, v in state_dict.items()
        if not any(k.startswith(p) for p in _OBSOLETE_CHECKPOINT_PREFIXES)
    }


def is_rw_reset_key(name: str) -> bool:
    """Return True if ``name`` should be skipped when merging an init checkpoint (keep model init)."""
    if name in _LS_EXACT:
        return True
    for p in _RW_STATE_DICT_PREFIXES:
        if p.endswith("."):
            if name.startswith(p):
                return True
        elif name == p or name.startswith(p + "."):
            return True
    return False


def rw_parameter_keys(state_dict: Mapping[str, object]) -> Set[str]:
    """Keys in ``state_dict`` that are tensors and match :func:`is_rw_reset_key`."""
    return {k for k, v in state_dict.items() if isinstance(v, torch.Tensor) and is_rw_reset_key(k)}


def _migrate_node_init_to_role_emb(
    model_state_dict: Mapping[str, torch.Tensor],
    checkpoint_state_dict: Mapping[str, torch.Tensor],
) -> Optional[torch.Tensor]:
    """Map legacy ``node_init`` to role row 0; rows 1–2 keep the model's fresh init."""
    if _NODE_INIT_KEY not in checkpoint_state_dict:
        return None
    if _NODE_ROLE_EMB_KEY not in model_state_dict:
        return None
    node_init = checkpoint_state_dict[_NODE_INIT_KEY]
    if not isinstance(node_init, torch.Tensor):
        return None
    role_emb = model_state_dict[_NODE_ROLE_EMB_KEY].clone()
    if node_init.shape != role_emb[0].shape:
        raise ValueError(
            f"Cannot migrate {_NODE_INIT_KEY!r} to {_NODE_ROLE_EMB_KEY}[0]: "
            f"model {tuple(role_emb[0].shape)} vs checkpoint {tuple(node_init.shape)}."
        )
    role_emb[0] = node_init
    return role_emb


def merge_init_checkpoint_state_dict(
    model_state_dict: Mapping[str, torch.Tensor],
    checkpoint_state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Build ``state_dict`` for ``load_state_dict`` when resetting RW parameters.

    - Omits every RW-reset key so the model keeps freshly constructed values.
    - Loads matching non-RW checkpoint tensors when shapes agree.
    - Migrates legacy ``node_init`` to ``node_role_emb.weight[0]`` (rows 1–2 stay at
      model init) when the checkpoint predates role-based node embeddings.
    - Other model keys absent from the checkpoint are omitted so ``load_state_dict(...,
      strict=False)`` keeps freshly constructed values.

    Raises:
        ValueError: on non-RW shape mismatch for a checkpoint key that maps to the model.
    """
    filtered: Dict[str, torch.Tensor] = {}
    for k, v in checkpoint_state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if is_rw_reset_key(k):
            continue
        if k == _NODE_INIT_KEY:
            continue
        if k not in model_state_dict:
            continue
        mv = model_state_dict[k]
        if mv.shape != v.shape:
            raise ValueError(
                f"Non-RW tensor {k!r} shape mismatch: model {tuple(mv.shape)} vs checkpoint "
                f"{tuple(v.shape)}. Align ``--wander_*`` / architecture with the checkpoint, or only "
                "change RW-related flags together with ``--reset_rw_parameters``."
            )
        filtered[k] = v

    if _NODE_ROLE_EMB_KEY not in filtered:
        migrated = _migrate_node_init_to_role_emb(model_state_dict, checkpoint_state_dict)
        if migrated is not None:
            filtered[_NODE_ROLE_EMB_KEY] = migrated
            warnings.warn(
                f"Migrated checkpoint {_NODE_INIT_KEY!r} to {_NODE_ROLE_EMB_KEY}[0]; "
                f"rows 1-2 use fresh model initialization.",
                stacklevel=2,
            )

    missing: list[str] = []
    for k in model_state_dict:
        if is_rw_reset_key(k):
            continue
        if k not in filtered:
            missing.append(k)
    if missing:
        preview = ", ".join(missing[:40])
        more = f" (+{len(missing) - 40} more)" if len(missing) > 40 else ""
        warnings.warn(
            "Checkpoint missing non-RW keys; keeping fresh model init for: "
            f"{preview}{more}",
            stacklevel=2,
        )
    return filtered
