"""Registry mapping between Wander datasets and GraphPFN ICL evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SplitKind = Literal["external_npz", "graphland_rl"]


@dataclass(frozen=True)
class GraphPFNDatasetSpec:
    registry_key: str
    slug: str
    data_name: str
    is_graphland: bool
    is_binary: bool
    split_kind: SplitKind
    n_members: int = 10
    export_homogeneous: bool = False
    slurm_mem: str = "64G"
    export_data_dir: str = "data"
    ignore_features: bool = False
    relation_index: int | None = None


# Full Pokec is used only to build POKEC_REGIONS_100K_TOP10.
NC_EVAL_EXCLUDED_REGISTRY_KEYS: frozenset[str] = frozenset(
    {
        "POKEC_REGIONS",
    }
)


_GRAPHPFN_DATASET_SPECS_EXPLICIT: tuple[GraphPFNDatasetSpec, ...] = (
    GraphPFNDatasetSpec("ACTOR", "actor", "Actor", False, False, "external_npz"),
    GraphPFNDatasetSpec("AMAZON_RATINGS", "amazon-ratings", "Amazon-ratings", False, False, "external_npz"),
    GraphPFNDatasetSpec("BRAZIL", "brazil", "Brazil", False, False, "external_npz"),
    GraphPFNDatasetSpec(
        "CHAMELEON_FILTERED",
        "chameleon-filtered",
        "ChameleonFiltered",
        False,
        False,
        "external_npz",
        export_homogeneous=True,
    ),
    GraphPFNDatasetSpec("CITESEER", "citeseer", "CiteSeer", False, False, "external_npz"),
    GraphPFNDatasetSpec("CO_CS", "coauthor-cs", "Co_CS", False, False, "external_npz"),
    GraphPFNDatasetSpec(
        "CO_CS",
        "coauthor-cs-nofeat",
        "Co_CS",
        False,
        False,
        "external_npz",
        export_homogeneous=True,
        ignore_features=True,
    ),
    GraphPFNDatasetSpec("CO_PHYSICS", "coauthor-physics", "Co_Physics", False, False, "external_npz"),
    GraphPFNDatasetSpec(
        "CO_PHYSICS",
        "coauthor-physics-nofeat",
        "Co_Physics",
        False,
        False,
        "external_npz",
        export_homogeneous=True,
        ignore_features=True,
    ),
    GraphPFNDatasetSpec("COMPUTERS", "amazon-computers", "Computers", False, False, "external_npz"),
    GraphPFNDatasetSpec(
        "COMPUTERS",
        "amazon-computers-nofeat",
        "Computers",
        False,
        False,
        "external_npz",
        export_homogeneous=True,
        ignore_features=True,
    ),
    GraphPFNDatasetSpec("CORA", "cora", "Cora", False, False, "external_npz"),
    GraphPFNDatasetSpec("CORNELL", "cornell", "Cornell", False, False, "external_npz"),
    GraphPFNDatasetSpec("EUROPE", "europe", "Europe", False, False, "external_npz"),
    GraphPFNDatasetSpec("FULL_CORA", "full-cora", "Full_Cora", False, False, "external_npz"),
    GraphPFNDatasetSpec("FULL_DBLP", "full-dblp", "Full_DBLP", False, False, "external_npz"),
    GraphPFNDatasetSpec(
        "FULL_DBLP",
        "full-dblp-nofeat",
        "Full_DBLP",
        False,
        False,
        "external_npz",
        export_homogeneous=True,
        ignore_features=True,
    ),
    GraphPFNDatasetSpec("HM_CATEGORIES", "hm-categories", "hm-categories", True, False, "graphland_rl"),
    GraphPFNDatasetSpec("PHOTO", "amazon-photo", "Photo", False, False, "external_npz"),
    GraphPFNDatasetSpec(
        "PHOTO",
        "amazon-photo-nofeat",
        "Photo",
        False,
        False,
        "external_npz",
        export_homogeneous=True,
        ignore_features=True,
    ),
    GraphPFNDatasetSpec(
        "POKEC_REGIONS", "pokec-regions", "pokec-regions", True, False, "graphland_rl", 21
    ),
    GraphPFNDatasetSpec(
        "POKEC_REGIONS_100K_TOP10",
        "pokec-regions-100k-top10",
        "pokec-regions-100k-top10",
        True,
        False,
        "graphland_rl",
        export_homogeneous=True,
        slurm_mem="128G",
        export_data_dir="Wander/raw_data",
    ),
    GraphPFNDatasetSpec(
        "CITY_PARIS",
        "city-paris",
        "City-Paris",
        False,
        False,
        "external_npz",
        slurm_mem="256G",
    ),
    GraphPFNDatasetSpec(
        "CITY_SHANGHAI",
        "city-shanghai",
        "City-Shanghai",
        False,
        False,
        "external_npz",
        slurm_mem="256G",
    ),
    GraphPFNDatasetSpec(
        "CITY_LA",
        "city-la",
        "City-LA",
        False,
        False,
        "external_npz",
        slurm_mem="256G",
    ),
    GraphPFNDatasetSpec("PUBMED", "pubmed", "PubMed", False, False, "external_npz"),
    GraphPFNDatasetSpec(
        "PUBMED",
        "pubmed-nofeat",
        "PubMed",
        False,
        False,
        "external_npz",
        export_homogeneous=True,
        ignore_features=True,
    ),
    GraphPFNDatasetSpec("ROMAN_EMPIRE", "roman-empire", "Roman-empire", False, False, "external_npz"),
    GraphPFNDatasetSpec(
        "SQUIRREL_FILTERED",
        "squirrel-filtered",
        "SquirrelFiltered",
        False,
        False,
        "external_npz",
        export_homogeneous=True,
    ),
    GraphPFNDatasetSpec("TEXAS", "texas", "Texas", False, False, "external_npz"),
    GraphPFNDatasetSpec("USA", "usa", "USA", False, False, "external_npz"),
    GraphPFNDatasetSpec(
        "CYCLES",
        "cycles",
        "Cycles",
        False,
        False,
        "external_npz",
        export_homogeneous=True,
        export_data_dir="Wander/raw_data",
        ignore_features=True,
    ),
    GraphPFNDatasetSpec(
        "GRIDS",
        "grids",
        "Grids",
        False,
        False,
        "external_npz",
        export_homogeneous=True,
        export_data_dir="Wander/raw_data",
        ignore_features=True,
    ),
    GraphPFNDatasetSpec("WIKI_CS", "wiki-cs", "Wiki_CS", False, False, "external_npz"),
    GraphPFNDatasetSpec("WISCONSIN", "wisconsin", "Wisconsin", False, False, "external_npz"),
)

_LARGE_NOFEAT_SLUGS = frozenset(
    {
        "pokec-regions",
        "pokec-regions-100k-top10",
        "city-paris",
        "city-shanghai",
        "city-la",
        "hm-categories",
        "full-cora",
    }
)


def _nofeat_spec(spec: GraphPFNDatasetSpec) -> GraphPFNDatasetSpec:
    """Homogeneous npy export with empty features + GraphPFN random features."""
    mem = "128G" if (spec.slug in _LARGE_NOFEAT_SLUGS or spec.slurm_mem == "128G") else spec.slurm_mem
    return GraphPFNDatasetSpec(
        registry_key=spec.registry_key,
        slug=f"{spec.slug}-nofeat",
        data_name=spec.data_name,
        is_graphland=spec.is_graphland,
        is_binary=spec.is_binary,
        # Exported npy bundles always use external split.npz (RL masks baked in for GraphLand).
        split_kind="external_npz",
        n_members=spec.n_members,
        export_homogeneous=True,
        slurm_mem=mem,
        export_data_dir=spec.export_data_dir,
        ignore_features=True,
        relation_index=spec.relation_index,
    )


def _with_auto_nofeat(specs: tuple[GraphPFNDatasetSpec, ...]) -> tuple[GraphPFNDatasetSpec, ...]:
    existing = {s.slug for s in specs}
    extra: list[GraphPFNDatasetSpec] = []
    for spec in specs:
        if spec.ignore_features:
            continue
        nf = _nofeat_spec(spec)
        if nf.slug in existing:
            continue
        extra.append(nf)
        existing.add(nf.slug)
    return specs + tuple(extra)


GRAPHPFN_DATASET_SPECS: tuple[GraphPFNDatasetSpec, ...] = _with_auto_nofeat(
    _GRAPHPFN_DATASET_SPECS_EXPLICIT
)


def graphpfn_slugs() -> list[str]:
    return [spec.slug for spec in GRAPHPFN_DATASET_SPECS]


def export_homogeneous_slugs() -> list[str]:
    return [spec.slug for spec in GRAPHPFN_DATASET_SPECS if spec.export_homogeneous]


def ignore_features_slugs() -> list[str]:
    return [spec.slug for spec in GRAPHPFN_DATASET_SPECS if spec.ignore_features]


def spec_by_slug(slug: str) -> GraphPFNDatasetSpec:
    for spec in GRAPHPFN_DATASET_SPECS:
        if spec.slug == slug:
            return spec
    raise KeyError(f"Unknown GraphPFN dataset slug: {slug}")


def graphpfn_multiclass_nc_slugs(*, featured_only: bool = True) -> list[str]:
    """Featured (or all) multiclass GraphPFN slugs."""
    excluded_keys = NC_EVAL_EXCLUDED_REGISTRY_KEYS
    slugs: list[str] = []
    for spec in GRAPHPFN_DATASET_SPECS:
        if spec.registry_key in excluded_keys:
            continue
        if spec.is_binary:
            continue
        if featured_only and spec.ignore_features:
            continue
        slugs.append(spec.slug)
    return slugs


def spec_by_registry_key(registry_key: str) -> GraphPFNDatasetSpec:
    key = registry_key.upper()
    for spec in GRAPHPFN_DATASET_SPECS:
        if spec.registry_key == key:
            return spec
    raise KeyError(f"Unknown GraphPFN registry key: {registry_key}")


def nc_eval_specs(nofeat: bool | None = None) -> tuple[GraphPFNDatasetSpec, ...]:
    """GraphPFN ICL specs allowed for GraphAny / AnyGraph node classification.

    Drops full Pokec. ``nofeat`` selects
    featured slugs (``False``), ``*-nofeat`` twins (``True``), or both (``None``).
    """
    specs = [
        spec
        for spec in GRAPHPFN_DATASET_SPECS
        if spec.registry_key not in NC_EVAL_EXCLUDED_REGISTRY_KEYS
    ]
    if nofeat is None:
        return tuple(specs)
    return tuple(spec for spec in specs if spec.ignore_features is nofeat)


def nc_eval_slugs(nofeat: bool | None = None) -> list[str]:
    return [spec.slug for spec in nc_eval_specs(nofeat=nofeat)]


def resolve_nc_eval_spec(name: str, *, nofeat: bool = False) -> GraphPFNDatasetSpec:
    """Resolve a GraphPFN slug or registry key to an NC-eval spec."""
    raw = name.strip()
    if not raw:
        raise KeyError("Empty dataset name")
    slug = raw.lower()
    if nofeat and not slug.endswith("-nofeat"):
        slug = f"{slug}-nofeat"
    try:
        spec = spec_by_slug(slug)
    except KeyError:
        spec = spec_by_registry_key(raw)
        if nofeat and not spec.ignore_features:
            spec = spec_by_slug(f"{spec.slug}-nofeat")
    allowed = {s.slug for s in nc_eval_specs()}
    if spec.slug not in allowed:
        raise KeyError(
            f"{spec.slug!r} is not in the GraphAny/AnyGraph NC eval set "
            "(excluded: full Pokec)"
        )
    return spec
