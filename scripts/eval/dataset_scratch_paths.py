#!/usr/bin/env python3
"""Print raw_data-relative paths required to load each dataset spec."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset import get_datasetargs
from data.datasets import DATASET_REGISTRY
from data.datasets.kg_eval.common import MTDEA_REGISTRY

GRAPHland_KEYS = frozenset(
    {
        "HM_CATEGORIES",
        "POKEC_REGIONS",
    }
)

def _anygraph_folder(registry_key: str) -> str | None:
    from data.datasets.anygraph_lp.register import _DATASETS, _registry_key

    for folder in _DATASETS:
        if _registry_key(folder) == registry_key:
            return folder
    return None


def scratch_paths_for_dataset(spec: str) -> list[str]:
    """Return paths relative to raw_data/ required to load ``spec`` (e.g. CORA, FB15K237_IND:v1)."""
    args = get_datasetargs(spec)
    key = args["registry_key"]
    version = args["dataset_version"]
    name = args["name"]
    paths: set[str] = set()

    if key in {"CYCLES", "GRIDS"}:
        paths.add(key)
        return sorted(paths)

    if key.startswith("ANYGRAPH_"):
        folder = _anygraph_folder(key)
        if folder is not None:
            paths.add(f"AnyGraph/zero-shot datasets/{folder}")
        return sorted(paths)

    if key.startswith("UNILP_"):
        from data.datasets.unilp_lp import UNILP_DIRNAME

        paths.add(UNILP_DIRNAME)
        return sorted(paths)

    if key in MTDEA_REGISTRY:
        paths.add(f"{key}/default")
        return sorted(paths)

    if key == "POKEC_REGIONS_100K_TOP10":
        paths.add("GraphLand/pokec-regions")
        paths.add("POKEC_REGIONS_100K")
        return sorted(paths)

    if key in GRAPHland_KEYS:
        paths.add(f"GraphLand/{name}")
        return sorted(paths)

    mod = DATASET_REGISTRY[key]
    if getattr(mod, "load_graph_bundle", None) is not None:
        if args["is_inductive"]:
            supported = args["supported_versions"]
            if supported:
                ver = version or supported[0]
                paths.add(f"{name}/{ver}")
            else:
                paths.add(f"{key}/default")
        else:
            paths.add(name)
        return sorted(paths)

    paths.add(name)
    return sorted(paths)


def scratch_paths_for_datasets(specs: list[str]) -> list[str]:
    out: set[str] = set()
    for spec in specs:
        if not spec.strip():
            continue
        out.update(scratch_paths_for_dataset(spec.strip()))
    return sorted(out)


def main() -> None:
    specs = sys.argv[1:]
    if not specs:
        return
    for rel in scratch_paths_for_datasets(specs):
        print(rel)


if __name__ == "__main__":
    main()
